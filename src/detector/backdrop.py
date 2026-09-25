"""Backdrop check: is the background behind a misread test 1.5 kg plate yellower?

Implements "Backdrop check" in NOTES.md, which was fixed before any of this ran. Constants
below are that protocol's numbers; changing one means writing a new protocol, not editing
this file.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from detector.evaluate import MISSED, Box, assign_image, load_predictions

TARGET = "1.5kg"
MISREAD_AS = "0.5kg"
CONFIDENCE_THRESHOLD = 0.80  # Run 1's, chosen on val; not picked again
MARGIN_FRACTION = 0.25
MARGIN_MIN_PX = 8
MIN_RING_PIXELS = 150
PLATE_CORE = 0.60  # central share of the box, each dimension, for the plate's own colour
BOOTSTRAP_REPS = 10_000
SEED = 20260925
MIN_BOXES, MIN_IMAGES = 15, 6

M, C = "misread", "correct"


# --- colour --------------------------------------------------------------------------------


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """(..., 3) uint8 sRGB → (..., 3) CIELAB, D65 white, standard formulas."""
    c = rgb.astype(np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    lab = np.empty_like(f)
    lab[..., 0] = 116 * f[..., 1] - 16
    lab[..., 1] = 500 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200 * (f[..., 1] - f[..., 2])
    return lab


# --- geometry ------------------------------------------------------------------------------


def _pixel_span(lo: float, hi: float, limit: int) -> tuple[int, int]:
    """Pixel indices whose centres fall inside [lo, hi), clipped to [0, limit)."""
    a = max(0, int(np.ceil(lo - 0.5)))
    b = min(limit, int(np.ceil(hi - 0.5)))
    return a, max(a, b)


def ring_mask(box: Box, all_boxes: list[Box], shape: tuple[int, int]) -> np.ndarray:
    """Pixels within the grown box, minus every labelled box in the image."""
    h_img, w_img = shape
    x, y, w, h = box
    m = max(MARGIN_MIN_PX, MARGIN_FRACTION * max(w, h))
    mask = np.zeros(shape, dtype=bool)
    x0, x1 = _pixel_span(x - m, x + w + m, w_img)
    y0, y1 = _pixel_span(y - m, y + h + m, h_img)
    mask[y0:y1, x0:x1] = True
    for bx, by, bw, bh in all_boxes:
        x0, x1 = _pixel_span(bx, bx + bw, w_img)
        y0, y1 = _pixel_span(by, by + bh, h_img)
        mask[y0:y1, x0:x1] = False
    return mask


def core_mask(box: Box, shape: tuple[int, int]) -> np.ndarray:
    x, y, w, h = box
    dx, dy = w * (1 - PLATE_CORE) / 2, h * (1 - PLATE_CORE) / 2
    mask = np.zeros(shape, dtype=bool)
    x0, x1 = _pixel_span(x + dx, x + w - dx, shape[1])
    y0, y1 = _pixel_span(y + dy, y + h - dy, shape[0])
    mask[y0:y1, x0:x1] = True
    return mask


@dataclass(frozen=True)
class BoxMeasure:
    image: str
    outcome: str  # predicted class, or MISSED
    ring_pixels: int
    ring_b: float | None  # None = unmeasurable
    plate_chroma: float | None


def measure_box(rgb: np.ndarray, lab: np.ndarray, box: Box, all_boxes: list[Box]) -> tuple:
    ring = ring_mask(box, all_boxes, rgb.shape[:2]) & (rgb.max(axis=2) > 0)
    n = int(ring.sum())
    ring_b = float(np.median(lab[..., 2][ring])) if n >= MIN_RING_PIXELS else None
    core = core_mask(box, rgb.shape[:2])
    chroma = None
    if core.any():
        a, b = lab[..., 1][core], lab[..., 2][core]
        chroma = float(np.median(np.hypot(a, b)))
    return n, ring_b, chroma


def measure_split(split_dir: Path, predictions_csv: Path) -> list[BoxMeasure]:
    """Measure every ground-truth TARGET box in a prepared split."""
    coco = json.loads((split_dir / "_annotations.coco.json").read_text())
    names = {c["id"]: c["name"] for c in coco["categories"]}
    per_image: dict[int, list[tuple[str, Box]]] = {im["id"]: [] for im in coco["images"]}
    for a in coco["annotations"]:
        per_image[a["image_id"]].append((names[a["category_id"]], tuple(a["bbox"])))
    preds = load_predictions(predictions_csv)

    out: list[BoxMeasure] = []
    for im in coco["images"]:
        gt = per_image[im["id"]]
        if not any(c == TARGET for c, _ in gt):
            continue
        original = im.get("extra", {}).get("name", im["file_name"])
        rgb = np.asarray(Image.open(split_dir / im["file_name"]).convert("RGB"))
        if rgb.shape[:2] != (im["height"], im["width"]):
            raise ValueError(
                f"{im['file_name']}: pixels {rgb.shape[1]}x{rgb.shape[0]} != COCO "
                f"{im['width']}x{im['height']} - boxes would not line up"
            )
        lab = srgb_to_lab(rgb)
        outcome, _ = assign_image(gt, preds.get(original, []), CONFIDENCE_THRESHOLD)
        all_boxes = [b for _, b in gt]
        for (cls, box), o in zip(gt, outcome, strict=True):
            if cls == TARGET:
                n, ring_b, chroma = measure_box(rgb, lab, box, all_boxes)
                out.append(BoxMeasure(original, o, n, ring_b, chroma))
    return out


# --- summary -------------------------------------------------------------------------------


def group_of(outcome: str) -> str:
    if outcome == MISREAD_AS:
        return M
    if outcome == TARGET:
        return C
    return MISSED if outcome == MISSED else "other"


def auc(pos: list[float], neg: list[float]) -> float:
    """P(pos > neg), ties ½ — the Mann-Whitney AUC."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def cluster_bootstrap_auc(
    by_image: dict[str, tuple[list[float], list[float]]],
    reps: int = BOOTSTRAP_REPS,
    seed: int = SEED,
) -> tuple[tuple[float, float], int]:
    """95% percentile CI for AUC, resampling whole images. Returns (CI, skipped resamples)."""
    rng = random.Random(seed)
    images = sorted(by_image)
    stats, skipped = [], 0
    for _ in range(reps):
        pos: list[float] = []
        neg: list[float] = []
        for name in rng.choices(images, k=len(images)):
            p, n = by_image[name]
            pos += p
            neg += n
        if not pos or not neg:
            skipped += 1
            continue
        stats.append(auc(pos, neg))
    lo, hi = np.percentile(stats, [2.5, 97.5]) if stats else (float("nan"), float("nan"))
    return (float(lo), float(hi)), skipped


def decide(n_boxes: dict[str, int], n_images: dict[str, int], a: float, ci: tuple) -> str:
    """The pre-registered rule: for / reversed / against / inconclusive."""
    if any(n_boxes[g] < MIN_BOXES or n_images[g] < MIN_IMAGES for g in (M, C)):
        return "inconclusive"
    lo, hi = ci
    if a >= 0.70 and lo > 0.55:
        return "for"
    if a <= 0.30 and hi < 0.45:
        return "reversed"
    if lo >= 0.30 and hi <= 0.70:
        return "against"
    return "inconclusive"


def _describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return {"n": len(values), "median": float(med), "q1": float(q1), "q3": float(q3)}


def summarise(measures: list[BoxMeasure], reps: int = BOOTSTRAP_REPS) -> dict:
    groups = sorted({group_of(m.outcome) for m in measures} | {M, C})
    ok = [m for m in measures if m.ring_b is not None]
    ring = {g: [m.ring_b for m in ok if group_of(m.outcome) == g] for g in groups}
    chroma = {
        g: [m.plate_chroma for m in ok if group_of(m.outcome) == g and m.plate_chroma is not None]
        for g in groups
    }
    n_boxes = {g: len(ring[g]) for g in groups}
    n_images = {g: len({m.image for m in ok if group_of(m.outcome) == g}) for g in groups}
    unmeasurable = {
        g: sum(1 for m in measures if m.ring_b is None and group_of(m.outcome) == g) for g in groups
    }

    by_image: dict[str, tuple[list[float], list[float]]] = {}
    for m in ok:
        g = group_of(m.outcome)
        if g in (M, C):
            pos, neg = by_image.setdefault(m.image, ([], []))
            (pos if g == M else neg).append(m.ring_b)

    a = auc(ring[M], ring[C])
    ci, skipped = cluster_bootstrap_auc(by_image, reps=reps)
    both = {k: v for k, v in by_image.items() if v[0] and v[1]}
    within_wins = sum(
        (p > q) + 0.5 * (p == q) for pos, neg in both.values() for p in pos for q in neg
    )
    within_pairs = sum(len(pos) * len(neg) for pos, neg in both.values())

    return {
        "protocol": "NOTES.md - Backdrop check",
        "target": TARGET,
        "misread_as": MISREAD_AS,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "n_target_boxes": len(measures),
        "measurable_boxes": n_boxes,
        "images": n_images,
        "unmeasurable_boxes": unmeasurable,
        "ring_b": {g: _describe(v) for g, v in ring.items()},
        "plate_chroma_secondary": {g: _describe(v) for g, v in chroma.items()},
        "auc_misread_vs_correct": a,
        "auc_ci95_cluster_bootstrap": list(ci),
        "bootstrap_reps": reps,
        "bootstrap_seed": SEED,
        "bootstrap_skipped": skipped,
        "within_image": {
            "images_with_both": len(both),
            "pairs": within_pairs,
            "auc": within_wins / within_pairs if within_pairs else None,
        },
        "verdict": decide(n_boxes, n_images, a, ci),
    }


def format_summary(s: dict) -> str:
    lo, hi = s["auc_ci95_cluster_bootstrap"]
    lines = [
        f"Test {s['target']} boxes: {s['n_target_boxes']}; confidence threshold "
        f"{s['confidence_threshold']} (Run 1's, from val)",
        "",
        "| group | measurable boxes | images | unmeasurable | ring b* median (IQR) "
        "| plate C* median (IQR) |",
        "|---|--:|--:|--:|--:|--:|",
    ]

    def fmt(d: dict) -> str:
        return f"{d['median']:.1f} ({d['q1']:.1f} to {d['q3']:.1f})" if d["n"] else "-"

    for g in s["ring_b"]:
        lines.append(
            f"| {g} | {s['measurable_boxes'][g]} | {s['images'][g]} | "
            f"{s['unmeasurable_boxes'][g]} | {fmt(s['ring_b'][g])} | "
            f"{fmt(s['plate_chroma_secondary'][g])} |"
        )
    w = s["within_image"]
    within = f"{w['auc']:.3f}" if w["auc"] is not None else "-"
    lines += [
        "",
        f"AUC (ring b*, misread > correct): **{s['auc_misread_vs_correct']:.3f}**"
        f"; 95% CI {lo:.3f} to {hi:.3f} (cluster bootstrap over images, "
        f"{s['bootstrap_reps']} reps, seed {s['bootstrap_seed']}, "
        f"{s['bootstrap_skipped']} skipped)",
        f"Within-image: {w['images_with_both']} photos with both groups, {w['pairs']} pairs, "
        f"AUC {within}",
        "",
        f"**Verdict: {s['verdict']}**",
    ]
    return "\n".join(lines)


def measures_to_rows(measures: list[BoxMeasure]) -> list[dict]:
    return [asdict(m) | {"group": group_of(m.outcome)} for m in measures]
