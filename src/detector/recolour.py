"""Recolouring test: does the read change when only the backdrop colour changes?

Implements the transform in "Recolouring test" in NOTES.md — the target-colour rule, the
per-arm mask/target combinations, and the sRGB round trip. This module never runs the
detector; it only builds the altered images and the numbers the protocol asks the transform
itself to report (share of pixels clipped). The baseline gate and the one inference run are a
separate step, not implemented here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from detector.backdrop import MIN_RING_PIXELS, TARGET, _pixel_span, ring_mask, srgb_to_lab
from detector.evaluate import Box

RING, SHAM, WHOLE, REVERSE, KEEP = "ring", "sham", "whole", "reverse", "keep"
BLUE, YELLOW = "blue", "yellow"

# (mask kind, target name); target name None = sham (round trip, no colour change).
ARMS: dict[str, tuple[str, str | None]] = {
    RING: ("ring", BLUE),
    SHAM: ("ring", None),
    WHOLE: ("whole", BLUE),
    REVERSE: ("ring", YELLOW),
    KEEP: ("ring", BLUE),
}


# --- colour ----------------------------------------------------------------------------------

_M_INV = np.linalg.inv(
    np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
)
_WHITE = np.array([0.95047, 1.0, 1.08883])
_EPS, _KAPPA = 216 / 24389, 24389 / 27
_DELTA = (_EPS) ** (1 / 3)  # 6/29


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """(..., 3) CIELAB, D65 white → (..., 3) sRGB in [0, 255], float, NOT clipped or rounded.

    Exact inverse of ``srgb_to_lab`` (same matrix, white point and thresholds).
    """
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    xr = np.where(fx > _DELTA, fx**3, (fx - 16 / 116) * 3 * _DELTA**2)
    zr = np.where(fz > _DELTA, fz**3, (fz - 16 / 116) * 3 * _DELTA**2)
    yr = np.where(lab[..., 0] > _KAPPA * _EPS, fy**3, lab[..., 0] / _KAPPA)
    xyz = np.stack([xr, yr, zr], axis=-1) * _WHITE
    lin = xyz @ _M_INV.T
    c = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.clip(lin, 0, None) ** (1 / 2.4) - 0.055)
    return c * 255.0


# --- masks -------------------------------------------------------------------------------------


def whole_mask(all_boxes: list[Box], shape: tuple[int, int], rgb: np.ndarray) -> np.ndarray:
    """Every pixel outside every labelled box in the image, minus pure-black padding."""
    mask = np.ones(shape, dtype=bool)
    for bx, by, bw, bh in all_boxes:
        x0, x1 = _pixel_span(bx, bx + bw, shape[1])
        y0, y1 = _pixel_span(by, by + bh, shape[0])
        mask[y0:y1, x0:x1] = False
    return mask & (rgb.max(axis=2) > 0)


# --- transform ---------------------------------------------------------------------------------


def apply_recolour(
    rgb: np.ndarray, mask: np.ndarray, target_ab: tuple[float, float] | None
) -> tuple[np.ndarray, float]:
    """Keep L*, set every masked pixel's (a*, b*) to ``target_ab``.

    ``target_ab=None`` is the sham: no colour change, just the sRGB -> Lab -> sRGB round trip,
    to measure the pipeline's own noise floor. Pixels outside ``mask`` are copied unchanged
    (bit-identical to ``rgb``), never round-tripped. Returns the recoloured uint8 image and the
    share of masked pixels whose Lab value fell outside the sRGB gamut and had to be clipped.
    """
    lab = srgb_to_lab(rgb)
    if target_ab is not None:
        lab = lab.copy()
        lab[..., 1] = target_ab[0]
        lab[..., 2] = target_ab[1]
    raw = lab_to_srgb(lab)
    out_of_gamut = (raw < 0) | (raw > 255)
    rounded = np.clip(np.round(raw), 0, 255).astype(np.uint8)
    out = rgb.copy()
    out[mask] = rounded[mask]
    n = int(mask.sum())
    clipped = float(out_of_gamut[mask].any(axis=-1).sum() / n) if n else 0.0
    return out, clipped


def apply_arm(
    rgb: np.ndarray,
    box: Box,
    all_boxes: list[Box],
    arm: str,
    targets: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, float]:
    """Build one arm's altered image for one box, per the ``ARMS`` table."""
    mask_kind, target_name = ARMS[arm]
    shape = rgb.shape[:2]
    if mask_kind == "ring":
        mask = ring_mask(box, all_boxes, shape) & (rgb.max(axis=2) > 0)
    else:
        mask = whole_mask(all_boxes, shape, rgb)
    target_ab = targets[target_name] if target_name is not None else None
    return apply_recolour(rgb, mask, target_ab)


# --- target colours ------------------------------------------------------------------------


@dataclass(frozen=True)
class RingColour:
    image: str
    ring_pixels: int
    ring_a: float | None
    ring_b: float | None


def measure_ring_ab(split_dir: Path, target_cls: str = TARGET) -> list[RingColour]:
    """Median ring a*, b* for every ``target_cls`` box in a prepared split. Outcome-blind — the
    target rule uses every box on a side, whatever it was read as."""
    coco = json.loads((split_dir / "_annotations.coco.json").read_text())
    names = {c["id"]: c["name"] for c in coco["categories"]}
    per_image: dict[int, list[tuple[str, Box]]] = {im["id"]: [] for im in coco["images"]}
    for a in coco["annotations"]:
        per_image[a["image_id"]].append((names[a["category_id"]], tuple(a["bbox"])))

    out: list[RingColour] = []
    for im in coco["images"]:
        gt = per_image[im["id"]]
        boxes = [b for c, b in gt if c == target_cls]
        if not boxes:
            continue
        original = im.get("extra", {}).get("name", im["file_name"])
        rgb = np.asarray(Image.open(split_dir / im["file_name"]).convert("RGB"))
        if rgb.shape[:2] != (im["height"], im["width"]):
            raise ValueError(
                f"{im['file_name']}: pixels {rgb.shape[1]}x{rgb.shape[0]} != COCO "
                f"{im['width']}x{im['height']} - boxes would not line up"
            )
        lab = srgb_to_lab(rgb)
        all_boxes = [b for _, b in gt]
        for box in boxes:
            ring = ring_mask(box, all_boxes, rgb.shape[:2]) & (rgb.max(axis=2) > 0)
            n = int(ring.sum())
            if n >= MIN_RING_PIXELS:
                a_med = float(np.median(lab[..., 1][ring]))
                b_med = float(np.median(lab[..., 2][ring]))
            else:
                a_med = b_med = None
            out.append(RingColour(original, n, a_med, b_med))
    return out


def target_colours(rows: list[RingColour]) -> dict:
    """Blue/yellow target (a*, b*), fixed by rule: median-of-medians of every measurable box's
    ring, split by the sign of that box's own ring b* (<=0 blue side, >0 yellow side)."""
    ok = [r for r in rows if r.ring_b is not None]
    blue = [r for r in ok if r.ring_b <= 0]
    yellow = [r for r in ok if r.ring_b > 0]

    def med(rs: list[RingColour], attr: str) -> float:
        return float(np.median([getattr(r, attr) for r in rs])) if rs else float("nan")

    return {
        "blue": (med(blue, "ring_a"), med(blue, "ring_b")),
        "yellow": (med(yellow, "ring_a"), med(yellow, "ring_b")),
        "n_blue": len(blue),
        "n_yellow": len(yellow),
        "n_unmeasurable": len(rows) - len(ok),
    }
