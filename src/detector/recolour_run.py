"""Baseline gate and the five-arm run for the recolouring test (NOTES.md).

This is the only module in the recolouring test that calls the model. Order, per the
protocol: baseline gate first, on unaltered images, over every box in the arms (M +
C-blue + C-yellow, from ``results/backdrop_boxes.csv``). More than 2 mismatches against
Run 1's own outcomes stops the run before any image is altered. If the gate passes, the
five arms run once each, using Run 1's matching unchanged (class-agnostic greedy, IoU 0.5,
confidence threshold 0.80).

``predict_fn`` is injected everywhere a prediction is needed, so the gate/arm logic is
tested without a model (see ``tests/test_recolour_run.py``); only ``load_model``/``predict``
touch ``rfdetr`` itself.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from detector.backdrop import MISREAD_AS as _MISREAD_AS
from detector.backdrop import TARGET as _TARGET
from detector.backdrop import C, M
from detector.evaluate import Box, Detection, assign_image, iou, load_ground_truth, wilson
from detector.recolour import (
    KEEP,
    REVERSE,
    RING,
    SHAM,
    WHOLE,
    apply_arm,
    measure_ring_ab,
    target_colours,
)

CONFIDENCE_THRESHOLD = 0.80  # Run 1's, chosen on val; not picked again
PRED_THRESHOLD = 0.01  # keep every raw detection; assign_image applies the real cut
MAX_BASELINE_MISMATCH = 2
PredictFn = Callable[[np.ndarray], list[Detection]]


# --- model -----------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(checkpoint: Path, device: str = "cpu") -> Any:
    from rfdetr import RFDETR

    return RFDETR.from_checkpoint(str(checkpoint), trust_checkpoint=True, device=device)


def predict(model: Any, rgb: np.ndarray) -> list[Detection]:
    det = model.predict(Image.fromarray(rgb), threshold=PRED_THRESHOLD)
    names = det.data["class_name"] if det.data else []
    out = []
    for (x0, y0, x1, y1), conf, name in zip(det.xyxy, det.confidence, names, strict=True):
        box = (float(x0), float(y0), float(x1 - x0), float(y1 - y0))
        out.append(Detection(str(name), float(conf), box))
    return out


# --- data access -------------------------------------------------------------------------------


def name_to_file(coco: dict) -> dict[str, str]:
    return {
        im.get("extra", {}).get("name", im["file_name"]): im["file_name"] for im in coco["images"]
    }


def load_rgb(split_dir: Path, coco: dict, image: str) -> np.ndarray:
    return np.asarray(Image.open(split_dir / name_to_file(coco)[image]).convert("RGB"))


def target_index(boxes: list[tuple[str, Box]]) -> int:
    """Index of the one ``1.5kg`` ground-truth box in an image's box list."""
    return next(i for i, (cls, _) in enumerate(boxes) if cls == _TARGET)


@dataclass(frozen=True)
class BackdropRow:
    image: str
    outcome: str
    group: str
    ring_b: float | None


def read_backdrop_rows(path: Path) -> list[BackdropRow]:
    rows = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            ring_b = float(r["ring_b"]) if r["ring_b"] else None
            rows.append(BackdropRow(r["image"], r["outcome"], r["group"], ring_b))
    return rows


# --- baseline gate -------------------------------------------------------------------------


@dataclass(frozen=True)
class GateCheck:
    image: str
    group: str
    recorded_outcome: str
    local_outcome: str

    @property
    def ok(self) -> bool:
        return self.local_outcome == self.recorded_outcome


def run_baseline_gate(
    predict_fn: PredictFn,
    split_dir: Path,
    coco: dict,
    gt: dict[str, list[tuple[str, Box]]],
    backdrop_rows: list[BackdropRow],
) -> list[GateCheck]:
    """Re-run Run 1's matching locally, on unaltered images, for every arm box (M + C)."""
    checks = []
    for row in backdrop_rows:
        if row.group not in (M, C):
            continue  # missed / other-class boxes are in no arm
        boxes = gt[row.image]
        idx = target_index(boxes)
        rgb = load_rgb(split_dir, coco, row.image)
        outcome, _ = assign_image(boxes, predict_fn(rgb), CONFIDENCE_THRESHOLD)
        checks.append(GateCheck(row.image, row.group, row.outcome, outcome[idx]))
    return checks


def select_arm_boxes(
    backdrop_rows: list[BackdropRow], gate: list[GateCheck]
) -> dict[str, list[str]]:
    """Images for each arm: M for Ring/Sham/Whole, C-blue for Reverse, C-yellow for Keep —
    dropping any box that failed the baseline gate."""
    ok = {c.image for c in gate if c.ok}
    correct_ok = [
        r for r in backdrop_rows if r.group == C and r.ring_b is not None and r.image in ok
    ]
    m = [r.image for r in backdrop_rows if r.group == M and r.image in ok]
    c_blue = [r.image for r in correct_ok if r.ring_b <= 0]
    c_yellow = [r.image for r in correct_ok if r.ring_b > 0]
    return {RING: m, SHAM: m, WHOLE: m, REVERSE: c_blue, KEEP: c_yellow}


# --- arms ------------------------------------------------------------------------------------


def is_success(arm: str, new_outcome: str) -> bool:
    """Ring/Sham/Whole/Keep want the box read as 1.5 kg; Reverse wants it read as 0.5 kg."""
    return new_outcome == (_MISREAD_AS if arm == REVERSE else _TARGET)


def best_iou_confidence(preds: list[Detection], box: Box, cls: str) -> float:
    """Highest confidence of any ``cls`` prediction with IoU >= 0.5 to ``box``, at any
    confidence (no threshold cut) — the secondary, non-decisive metric."""
    hits = [p.confidence for p in preds if p.cls == cls and iou(p.box, box) >= 0.5]
    return max(hits) if hits else 0.0


@dataclass(frozen=True)
class ArmOutcome:
    arm: str
    image: str
    new_outcome: str
    success: bool
    clipped_frac: float
    baseline_target_confidence: float
    new_target_confidence: float


def run_arm(
    predict_fn: PredictFn,
    split_dir: Path,
    coco: dict,
    gt: dict[str, list[tuple[str, Box]]],
    targets: dict[str, tuple[float, float]],
    arm: str,
    images: list[str],
    baseline_preds: dict[str, list[Detection]],
) -> list[ArmOutcome]:
    out = []
    for image in images:
        boxes = gt[image]
        idx = target_index(boxes)
        box = boxes[idx][1]
        all_boxes = [b for _, b in boxes]
        rgb = load_rgb(split_dir, coco, image)
        altered, clipped = apply_arm(rgb, box, all_boxes, arm, targets)
        preds = predict_fn(altered)
        outcome, _ = assign_image(boxes, preds, CONFIDENCE_THRESHOLD)
        new_outcome = outcome[idx]
        out.append(
            ArmOutcome(
                arm=arm,
                image=image,
                new_outcome=new_outcome,
                success=is_success(arm, new_outcome),
                clipped_frac=clipped,
                baseline_target_confidence=best_iou_confidence(baseline_preds[image], box, _TARGET),
                new_target_confidence=best_iou_confidence(preds, box, _TARGET),
            )
        )
    return out


# --- decision rule -------------------------------------------------------------------------


def decide(n: int, ring_flips: int, whole_flips: int, sham_flips: int) -> str:
    """Section 4 of the protocol. ``n`` is the (gate-passed) M-box count."""
    if sham_flips > 2:
        return "pipeline noise"
    if ring_flips >= n / 2:
        return "for (local)"
    if whole_flips >= n / 2:
        return "for (scene)"
    if ring_flips <= n / 10 and whole_flips <= n / 10:
        return "against"
    return "inconclusive"


# --- report ----------------------------------------------------------------------------------


def summarise(
    gate: list[GateCheck],
    arm_outcomes: dict[str, list[ArmOutcome]],
    targets: dict,
    checkpoint_sha256: str,
) -> dict:
    dropped = [c for c in gate if not c.ok]
    n = {arm: len(rows) for arm, rows in arm_outcomes.items()}
    flips = {arm: sum(r.success for r in rows) for arm, rows in arm_outcomes.items()}
    ci = {arm: wilson(flips[arm], n[arm]) for arm in arm_outcomes}
    conf_delta = {
        arm: sorted(r.new_target_confidence - r.baseline_target_confidence for r in rows)
        for arm, rows in arm_outcomes.items()
    }
    median_conf_delta = {arm: (float(np.median(d)) if d else None) for arm, d in conf_delta.items()}
    verdict = (
        decide(n[RING], flips[RING], flips[WHOLE], flips[SHAM])
        if all(a in arm_outcomes for a in (RING, WHOLE, SHAM))
        else "no verdict — baseline gate failed"
    )
    return {
        "protocol": "NOTES.md - Recolouring test",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "checkpoint_sha256": checkpoint_sha256,
        "platform": platform.platform(),
        "target_colours": {
            "blue": targets.get("blue"),
            "yellow": targets.get("yellow"),
            "n_blue": targets.get("n_blue"),
            "n_yellow": targets.get("n_yellow"),
        },
        "baseline_gate": {
            "n_checked": len(gate),
            "n_dropped": len(dropped),
            "dropped_images": [c.image for c in dropped],
            "passed": len(dropped) <= MAX_BASELINE_MISMATCH,
        },
        "arms": {
            arm: {
                "n": n[arm],
                "successes": flips[arm],
                "share": flips[arm] / n[arm] if n[arm] else None,
                "ci95": list(ci[arm]),
                "median_target_confidence_change": median_conf_delta[arm],
                "clipped_pixels_median": (
                    float(np.median([r.clipped_frac for r in arm_outcomes[arm]]))
                    if arm_outcomes[arm]
                    else None
                ),
            }
            for arm in arm_outcomes
        },
        "verdict": verdict,
    }


def format_summary(s: dict) -> str:
    g = s["baseline_gate"]
    lines = [
        f"Baseline gate: {g['n_checked']} boxes checked, {g['n_dropped']} dropped "
        f"({'passed' if g['passed'] else 'FAILED'}, max {MAX_BASELINE_MISMATCH})",
    ]
    if g["dropped_images"]:
        lines.append(f"Dropped: {', '.join(g['dropped_images'])}")
    tb, ty = s["target_colours"]["blue"], s["target_colours"]["yellow"]
    lines += [
        f"Targets (a*, b*): blue {tb} from {s['target_colours']['n_blue']} boxes, "
        f"yellow {ty} from {s['target_colours']['n_yellow']} boxes",
        "",
        "| arm | n | successes | share | 95% CI | median confidence change | median clipped |",
        "|---|--:|--:|--:|--:|--:|--:|",
    ]
    for arm, row in s["arms"].items():
        share = f"{row['share']:.1%}" if row["share"] is not None else "-"
        lo, hi = row["ci95"]
        mc = row["median_target_confidence_change"]
        mc_s = f"{mc:+.3f}" if mc is not None else "-"
        cl = row["clipped_pixels_median"]
        cl_s = f"{cl:.1%}" if cl is not None else "-"
        lines.append(
            f"| {arm} | {row['n']} | {row['successes']} | {share} | {lo:.1%} to {hi:.1%} "
            f"| {mc_s} | {cl_s} |"
        )
    lines += ["", f"**Verdict: {s['verdict']}**"]
    return "\n".join(lines)


def arm_rows(arm_outcomes: dict[str, list[ArmOutcome]]) -> list[dict]:
    return [asdict(r) for rows in arm_outcomes.values() for r in rows]


# --- orchestration -----------------------------------------------------------------------------


def run(
    prepared_dir: Path,
    results_dir: Path,
    checkpoint: Path,
    device: str = "cpu",
) -> tuple[dict, dict[str, list[ArmOutcome]]]:
    """Baseline gate, then (only if it passes) the five arms.

    Returns the report dict that ``format_summary`` renders (always includes the gate, even
    when it fails) and the per-box arm outcomes (empty when the gate fails).
    """
    test_dir = prepared_dir / "test"
    coco = json.loads((test_dir / "_annotations.coco.json").read_text())
    gt = load_ground_truth(test_dir / "_annotations.coco.json")
    backdrop_rows = read_backdrop_rows(results_dir / "backdrop_boxes.csv")

    model = load_model(checkpoint, device=device)
    checkpoint_sha256 = sha256_file(checkpoint)

    def predict_fn(rgb: np.ndarray) -> list[Detection]:
        return predict(model, rgb)

    gate = run_baseline_gate(predict_fn, test_dir, coco, gt, backdrop_rows)
    dropped = [c for c in gate if not c.ok]
    if len(dropped) > MAX_BASELINE_MISMATCH:
        report = summarise(gate, {}, {"blue": None, "yellow": None}, checkpoint_sha256)
        return report, {}

    arm_images = select_arm_boxes(backdrop_rows, gate)
    baseline_preds = {c.image: predict_fn(load_rgb(test_dir, coco, c.image)) for c in gate if c.ok}

    ring_rows = measure_ring_ab(test_dir)
    targets = target_colours(ring_rows)
    target_ab = {"blue": targets["blue"], "yellow": targets["yellow"]}

    arm_outcomes = {
        arm: run_arm(predict_fn, test_dir, coco, gt, target_ab, arm, images, baseline_preds)
        for arm, images in arm_images.items()
    }
    return summarise(gate, arm_outcomes, targets, checkpoint_sha256), arm_outcomes
