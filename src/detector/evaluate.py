"""Score saved predictions against ground truth: per-class P/R, confusions, same-colour pairs.

Implements the Run 1 evaluation protocol in NOTES.md. Works on the flat prediction CSVs the
training notebook writes, so none of this needs a GPU.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from detector.dataset import CLASSES

IOU_THRESHOLD = 0.5
THRESHOLD_GRID = [round(0.05 + 0.01 * i, 2) for i in range(91)]  # 0.05 … 0.95

# Plates of the same colour, (light, heavy). The model has to tell these apart by size.
COLOUR_PAIRS = {
    "red": ("2.5kg", "25kg"),
    "blue": ("2kg", "20kg"),
    "yellow": ("1.5kg", "15kg"),
    "green": ("1kg", "10kg"),
    "white": ("0.5kg", "5kg"),
}
PARTNER = {a: b for a, b in COLOUR_PAIRS.values()} | {b: a for a, b in COLOUR_PAIRS.values()}
HYPOTHESIS_CLASSES = ("0.5kg", "1.5kg")
CONTROL_CLASSES = ("1kg", "2kg", "2.5kg")

MISSED = "missed"
BACKGROUND = "background"

Box = tuple[float, float, float, float]  # x, y, w, h in pixels


@dataclass(frozen=True)
class Detection:
    cls: str
    confidence: float
    box: Box


def load_ground_truth(coco_path: Path) -> dict[str, list[tuple[str, Box]]]:
    """Ground-truth boxes per original image name, from a prepared split's COCO file."""
    coco = json.loads(coco_path.read_text())
    names = {c["id"]: c["name"] for c in coco["categories"]}
    by_id = {im["id"]: im.get("extra", {}).get("name", im["file_name"]) for im in coco["images"]}
    gt: dict[str, list[tuple[str, Box]]] = {name: [] for name in by_id.values()}
    for a in coco["annotations"]:
        gt[by_id[a["image_id"]]].append((names[a["category_id"]], tuple(a["bbox"])))
    return gt


def load_predictions(csv_path: Path) -> dict[str, list[Detection]]:
    preds: dict[str, list[Detection]] = defaultdict(list)
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            box = (float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]))
            preds[row["file"]].append(Detection(row["class_name"], float(row["confidence"]), box))
    return preds


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def match_image(
    gt: list[tuple[str, Box]], preds: list[Detection], threshold: float
) -> list[tuple[str, str]]:
    """Class-agnostic greedy matching for one image → (true class, predicted class) pairs.

    Unmatched ground truth comes back as (cls, MISSED); unmatched predictions as
    (BACKGROUND, cls).
    """
    kept = sorted((p for p in preds if p.confidence >= threshold), key=lambda p: -p.confidence)
    free = set(range(len(gt)))
    pairs: list[tuple[str, str]] = []
    for p in kept:
        best, best_iou = None, IOU_THRESHOLD
        for g in free:
            v = iou(p.box, gt[g][1])
            if v >= best_iou:
                best, best_iou = g, v
        if best is None:
            pairs.append((BACKGROUND, p.cls))
        else:
            free.remove(best)
            pairs.append((gt[best][0], p.cls))
    pairs.extend((gt[g][0], MISSED) for g in sorted(free))
    return pairs


def outcome_table(
    gt: dict[str, list[tuple[str, Box]]], preds: dict[str, list[Detection]], threshold: float
) -> Counter[tuple[str, str]]:
    """Counts of (true, predicted) over a whole split. Images missing from ``preds`` count as
    having no detections."""
    table: Counter[tuple[str, str]] = Counter()
    for name, boxes in gt.items():
        table.update(match_image(boxes, preds.get(name, []), threshold))
    return table


def micro_f1(table: Counter[tuple[str, str]]) -> float:
    correct = sum(n for (t, p), n in table.items() if t == p)
    predicted = sum(n for (t, p), n in table.items() if p != MISSED)
    actual = sum(n for (t, p), n in table.items() if t != BACKGROUND)
    return 2 * correct / (predicted + actual) if predicted + actual else 0.0


def pick_threshold(
    gt: dict[str, list[tuple[str, Box]]], preds: dict[str, list[Detection]]
) -> tuple[float, float]:
    """Confidence threshold with the best micro-F1 (lowest wins a tie). Call it on val only."""
    best_t, best_f1 = THRESHOLD_GRID[0], -1.0
    for t in THRESHOLD_GRID:
        f1 = micro_f1(outcome_table(gt, preds, t))
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, best_f1


def per_class(table: Counter[tuple[str, str]]) -> dict[str, dict[str, float | int]]:
    rows = {}
    for c in CLASSES:
        correct = table[(c, c)]
        predicted = sum(n for (t, p), n in table.items() if p == c)
        actual = sum(n for (t, p), n in table.items() if t == c)
        rows[c] = {
            "precision": correct / predicted if predicted else 0.0,
            "recall": correct / actual if actual else 0.0,
            "n_gt": actual,
            "n_pred": predicted,
        }
    return rows


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def error_breakdown(table: Counter[tuple[str, str]]) -> dict[str, dict]:
    """Per true class: correct / same-colour partner / other class / missed (sums to n_gt)."""
    rows = {}
    for c in CLASSES:
        n = sum(k for (t, p), k in table.items() if t == c)
        partner = PARTNER.get(c)
        counts = {
            "correct": table[(c, c)],
            "partner": table[(c, partner)] if partner else 0,
            "missed": table[(c, MISSED)],
        }
        counts["other"] = n - sum(counts.values())
        lo, hi = wilson(counts["partner"], n)
        rows[c] = {
            "partner_class": partner,
            "n_gt": n,
            **counts,
            "partner_share": counts["partner"] / n if n else 0.0,
            "partner_ci95": [lo, hi],
            "confused_as": {p: k for (t, p), k in table.items() if t == c and p not in (c, MISSED)},
        }
    return rows


def ap50(
    gt: dict[str, list[tuple[str, Box]]], preds: dict[str, list[Detection]]
) -> dict[str, float]:
    """Class-aware AP at IoU 0.5 with COCO's 101-point interpolation — the cross-check."""
    out = {}
    for c in CLASSES:
        n_gt = sum(1 for boxes in gt.values() for cls, _ in boxes if cls == c)
        dets = sorted(
            ((name, p) for name, ps in preds.items() for p in ps if p.cls == c),
            key=lambda x: -x[1].confidence,
        )
        used: dict[str, set[int]] = defaultdict(set)
        hits = []
        for name, p in dets:
            boxes = gt.get(name, [])
            best, best_iou = None, IOU_THRESHOLD
            for i, (cls, box) in enumerate(boxes):
                if cls == c and i not in used[name]:
                    v = iou(p.box, box)
                    if v >= best_iou:
                        best, best_iou = i, v
            if best is not None:
                used[name].add(best)
            hits.append(best is not None)
        if n_gt == 0:
            continue
        tp = fp = 0
        prec, rec = [], []
        for h in hits:
            tp, fp = tp + h, fp + (not h)
            prec.append(tp / (tp + fp))
            rec.append(tp / n_gt)
        # precision envelope, then sample at recall 0, 0.01, …, 1
        for i in range(len(prec) - 2, -1, -1):
            prec[i] = max(prec[i], prec[i + 1])
        samples = []
        for r in (i / 100 for i in range(101)):
            j = next((k for k, v in enumerate(rec) if v >= r), None)
            samples.append(prec[j] if j is not None else 0.0)
        out[c] = sum(samples) / 101
    return out


def verdict(val: dict[str, dict], test: dict[str, dict]) -> tuple[str, list[str]]:
    """Apply the pre-registered decision rule. Returns (supported|rejected|mixed, reasons)."""
    reasons, supported, missed_top = [], [], []
    for c in HYPOTHESIS_CLASSES:
        row = test[c]
        errors = {k: row[k] for k in ("partner", "other", "missed")}
        top = max(errors, key=errors.get)
        a = errors["partner"] > max(v for k, v in errors.items() if k != "partner")
        b = all(row["partner_share"] > test[k]["partner_share"] for k in CONTROL_CLASSES)
        c_ = row["partner_share"] > val[c]["partner_share"]
        supported.append(a and b and c_)
        missed_top.append(top == "missed" and errors["missed"] > errors["partner"])
        reasons.append(
            f"{c}: errors {errors}; (a) partner largest={a}, (b) above controls={b}, "
            f"(c) above own val share={c_}"
        )
    if all(supported):
        return "supported", reasons
    if all(missed_top):
        return "rejected", reasons
    return "mixed", reasons


def evaluate(prepared_dir: Path, results_dir: Path) -> dict:
    """Run the whole protocol and return a JSON-able report."""
    splits = {"val": "valid", "test": "test"}
    gt = {
        s: load_ground_truth(prepared_dir / d / "_annotations.coco.json") for s, d in splits.items()
    }
    preds = {s: load_predictions(results_dir / f"predictions_{s}.csv") for s in splits}

    threshold, val_f1 = pick_threshold(gt["val"], preds["val"])
    report: dict = {
        "protocol": "NOTES.md — Evaluation protocol for Run 1",
        "iou_threshold": IOU_THRESHOLD,
        "confidence_threshold": threshold,
        "val_micro_f1_at_threshold": val_f1,
    }
    breakdowns = {}
    for s in splits:
        table = outcome_table(gt[s], preds[s], threshold)
        breakdowns[s] = error_breakdown(table)
        report[s] = {
            "ap50_recomputed": ap50(gt[s], preds[s]),
            "micro_f1": micro_f1(table),
            "per_class": per_class(table),
            "errors": breakdowns[s],
            "background_predictions": sum(n for (t, _), n in table.items() if t == BACKGROUND),
        }
        aps = report[s]["ap50_recomputed"]
        report[s]["map50_recomputed"] = sum(aps.values()) / len(aps)
    report["verdict"], report["verdict_reasons"] = verdict(breakdowns["val"], breakdowns["test"])
    return report


def format_report(report: dict) -> str:
    lines = [
        f"IoU {report['iou_threshold']} · confidence threshold {report['confidence_threshold']} "
        f"(chosen on val, micro-F1 {report['val_micro_f1_at_threshold']:.3f})",
        f"mAP@50 recomputed: val {report['val']['map50_recomputed']:.3f} · "
        f"test {report['test']['map50_recomputed']:.3f}",
        "",
    ]
    for s in ("val", "test"):
        lines += [
            f"### {s}",
            "| true | n | correct | partner | other | missed | partner share (95% CI) | P | R |",
            "|---|--:|--:|--:|--:|--:|--:|--:|--:|",
        ]
        for c in CLASSES:
            e, pc = report[s]["errors"][c], report[s]["per_class"][c]
            partner = f"→{e['partner_class']} {e['partner']}" if e["partner_class"] else "—"
            lo, hi = e["partner_ci95"]
            share = (
                f"{e['partner_share']:.1%} ({lo:.1%} to {hi:.1%})" if e["partner_class"] else "—"
            )
            lines.append(
                f"| {c} | {e['n_gt']} | {e['correct']} | {partner} | {e['other']} | "
                f"{e['missed']} | {share} | {pc['precision']:.3f} | {pc['recall']:.3f} |"
            )
        lines += [f"background predictions: {report[s]['background_predictions']}", ""]
    lines.append(f"**Verdict: {report['verdict']}**")
    lines += [f"- {r}" for r in report["verdict_reasons"]]
    return "\n".join(lines)
