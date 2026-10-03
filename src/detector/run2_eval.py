"""Score Run 2 (2C control vs 2A colour cast) exactly as NOTES.md "Run 2" fixes it.

Written and tested before any Run 2 output exists, so the decision rule cannot be shaped by a
result. Reuses Run 1's matching and threshold rule unchanged (``detector.evaluate``).

Inputs, under one results directory: Run 1's files directly in it, and ``run2c/`` and ``run2a/``
each holding ``predictions_val.csv``, ``predictions_test.csv``, ``metrics.csv``,
``training_config.json`` and ``run_record.json``. Both arms must be present before a single
prediction file is opened, which enforces "both runs finish before either is evaluated".

Two interpretations the protocol leaves open, fixed here:

* **Val mAP@50-95 of the selected checkpoint (G2)** is the best value over epochs of
  ``val/mAP_50_95`` and ``val/ema_mAP_50_95``. ``rfdetr`` picks ``checkpoint_best_total`` on
  that metric, so its maximum is the selected checkpoint's value.
* **Guardrail boundaries** count as holding: G1/G2 hold at exactly -0.02 and G3 at exactly
  -0.05 (a tolerance of 1e-9 absorbs float noise in that direction only).
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from detector.dataset import CLASSES
from detector.dataset.prepare import OUTPUT_DIRS
from detector.evaluate import (
    MISSED,
    Box,
    Detection,
    assign_image,
    load_ground_truth,
    load_predictions,
    outcome_table,
    per_class,
    pick_threshold,
)
from detector.run2 import ARMS, Arm, config_problems

TARGET = "1.5kg"  # the class whose test boxes are the unit
MISREAD = "0.5kg"  # what it gets read as
OTHER = "other"
N_TARGET_BOXES = 100
ALPHA = 0.05
VOID_AT_OR_BELOW = 20  # M(2C) <= 20: the failure did not reproduce
MARGIN_MAP = 0.02  # G1, G2
MARGIN_CLASS_AP = 0.05  # G3
EPS = 1e-9
HYPOTHESIS = (MISREAD, TARGET)
OUTPUT_DIRS_USED = ("val", "test")  # train is never scored

RUN_FILES = (
    "predictions_val.csv",
    "predictions_test.csv",
    "metrics.csv",
    "training_config.json",
    "run_record.json",
)
SIDES = ("yellow", "blue")  # ring median b* > 0 / <= 0, Run 1's groups

STATES = ("void", "fixed", "fixed at a cost", "displaced", "not fixed")
CONSEQUENCE = {
    "void": "Run 1 stays. The 1.5 -> 0.5 kg failure did not reproduce in a retrain, so part of "
    "its size is run-to-run noise.",
    "fixed": "2A becomes the shipped checkpoint (HF weights and the demo).",
    "fixed at a cost": "Run 1 stays the shipped checkpoint. Reported as a mixed result.",
    "displaced": "Run 1 stays the shipped checkpoint. Misreads became misses, not correct reads.",
    "not fixed": "Run 1 stays the shipped checkpoint. Reported as a negative result.",
    "invalid": "No verdict: a run failed the config check or the runs are not comparable.",
}


# --- the primary metric -----------------------------------------------------------------------


def target_outcomes(
    gt: dict[str, list[tuple[str, Box]]], preds: dict[str, list[Detection]], threshold: float
) -> dict[str, str]:
    """Per image, what became of its one ground-truth 1.5 kg box: 1.5kg, 0.5kg, missed or other.

    Same matching as Run 1 (class-agnostic greedy, IoU 0.5, at ``threshold``).
    """
    out: dict[str, str] = {}
    for image, boxes in gt.items():
        idx = [i for i, (cls, _) in enumerate(boxes) if cls == TARGET]
        if not idx:
            continue
        if len(idx) > 1:
            raise ValueError(f"{image}: {len(idx)} ground-truth {TARGET} boxes, expected 1")
        got = assign_image(boxes, preds.get(image, []), threshold)[0][idx[0]]
        out[image] = got if got in (TARGET, MISREAD, MISSED) else OTHER
    return out


def mcnemar_one_sided(b: int, c: int) -> float:
    """P(X <= c) for X ~ Binomial(b + c, 1/2): exact, one-sided.

    ``b`` = boxes misread by the control and not by the treatment, ``c`` = the reverse. Small
    means the treatment misreads fewer.
    """
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(c + 1)) / 2**n


def paired_counts(control: dict[str, str], treatment: dict[str, str]) -> tuple[int, int]:
    if control.keys() != treatment.keys():
        raise ValueError("the two runs do not cover the same ground-truth boxes")
    b = sum(1 for i in control if control[i] == MISREAD and treatment[i] != MISREAD)
    c = sum(1 for i in control if treatment[i] == MISREAD and control[i] != MISREAD)
    return b, c


def count(outcomes: dict[str, str]) -> dict[str, int]:
    n = Counter(outcomes.values())
    return {k: n.get(k, 0) for k in (TARGET, MISREAD, MISSED, OTHER)}


# --- guardrails and the decision rule --------------------------------------------------------


@dataclass(frozen=True)
class Metrics:
    test_map_50_95: float
    test_map_50: float
    val_map_50_95: float  # best over epochs, see module docstring
    val_best_epoch: int
    test_ap: dict[str, float]  # per class, AP@50-95 (rfdetr's own)


def read_metrics(path: Path) -> Metrics:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    test_rows = [r for r in rows if r.get("test/mAP_50_95")]
    if len(test_rows) != 1:
        raise ValueError(f"{path}: expected one test row, found {len(test_rows)}")
    t = test_rows[0]
    best, best_epoch = -1.0, -1
    for r in rows:
        for col in ("val/mAP_50_95", "val/ema_mAP_50_95"):
            if r.get(col) and float(r[col]) > best:
                best, best_epoch = float(r[col]), int(float(r["epoch"]))
    if best < 0:
        raise ValueError(f"{path}: no validation mAP rows")
    return Metrics(
        test_map_50_95=float(t["test/mAP_50_95"]),
        test_map_50=float(t["test/mAP_50"]),
        val_map_50_95=best,
        val_best_epoch=best_epoch,
        test_ap={c: float(t[f"test/AP/{c}"]) for c in CLASSES},
    )


def guardrails(control: Metrics, treatment: Metrics) -> dict:
    """G1-G3 for the treatment against the control."""
    g1 = treatment.test_map_50_95 >= control.test_map_50_95 - MARGIN_MAP - EPS
    g2 = treatment.val_map_50_95 >= control.val_map_50_95 - MARGIN_MAP - EPS
    classes = {}
    for cls in CLASSES:
        if cls in HYPOTHESIS:
            continue
        drop = control.test_ap[cls] - treatment.test_ap[cls]
        classes[cls] = {
            "2C": control.test_ap[cls],
            "2A": treatment.test_ap[cls],
            "drop": drop,
            "ok": drop <= MARGIN_CLASS_AP + EPS,
        }
    return {
        "G1": {
            "2C": control.test_map_50_95,
            "2A": treatment.test_map_50_95,
            "margin": MARGIN_MAP,
            "ok": g1,
        },
        "G2": {
            "2C": control.val_map_50_95,
            "2A": treatment.val_map_50_95,
            "margin": MARGIN_MAP,
            "ok": g2,
        },
        "G3": {
            "margin": MARGIN_CLASS_AP,
            "classes": classes,
            "ok": all(v["ok"] for v in classes.values()),
        },
        "all_ok": g1 and g2 and all(v["ok"] for v in classes.values()),
    }


def decide(m_c: int, k_c: int, m_a: int, k_a: int, p: float, guardrails_ok: bool) -> tuple:
    """The Run 2 decision rule, in the protocol's order. Returns (state, criteria)."""
    criteria = {
        "void": m_c <= VOID_AT_OR_BELOW,
        "a_halved": m_a <= m_c // 2,
        "b_significant": p < ALPHA,
        "c_correct_not_missed": 2 * (k_a - k_c) >= (m_c - m_a),
        "d_guardrails": guardrails_ok,
    }
    a, b, c, d = (
        criteria["a_halved"],
        criteria["b_significant"],
        criteria["c_correct_not_missed"],
        criteria["d_guardrails"],
    )
    if criteria["void"]:
        return "void", criteria
    if a and b and c and d:
        return "fixed", criteria
    if a and b and c:
        return "fixed at a cost", criteria
    if a and b:
        return "displaced", criteria
    return "not fixed", criteria


# --- reading one run ---------------------------------------------------------------------------


def read_ring_b(path: Path) -> dict[str, float]:
    """Ring median b* per test 1.5 kg image, from Run 1's backdrop check."""
    with path.open(newline="") as f:
        return {r["image"]: float(r["ring_b"]) for r in csv.DictReader(f) if r["ring_b"]}


def _confusion(table: Counter[tuple[str, str]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for (true, pred), n in sorted(table.items()):
        out.setdefault(true, {})[pred] = n
    return out


def score_run(
    results_dir: Path,
    gt: dict[str, dict[str, list[tuple[str, Box]]]],
    ring_b: dict[str, float],
) -> dict:
    """Everything the protocol reports for one run, from its saved prediction and metric files."""
    preds = {s: load_predictions(results_dir / f"predictions_{s}.csv") for s in OUTPUT_DIRS_USED}
    threshold, val_f1 = pick_threshold(gt["val"], preds["val"])
    outcomes = target_outcomes(gt["test"], preds["test"], threshold)
    if len(outcomes) != N_TARGET_BOXES:
        raise ValueError(
            f"{len(outcomes)} test {TARGET} boxes, the protocol fixes {N_TARGET_BOXES}"
        )
    missing = outcomes.keys() - ring_b.keys()
    if missing:
        raise ValueError(f"no backdrop ring value for {sorted(missing)[:3]} ...")
    table = outcome_table(gt["test"], preds["test"], threshold)
    by_side = {
        side: count({i: o for i, o in outcomes.items() if (ring_b[i] > 0) == (side == "yellow")})
        for side in SIDES
    }
    metrics = read_metrics(results_dir / "metrics.csv")
    return {
        "threshold": threshold,
        "val_micro_f1": val_f1,
        "counts": count(outcomes),
        "by_side": by_side,
        "precision_0.5kg": per_class(table)[MISREAD]["precision"],
        "confusion": _confusion(table),
        "metrics": {
            "test_map_50_95": metrics.test_map_50_95,
            "test_map_50": metrics.test_map_50,
            "val_map_50_95": metrics.val_map_50_95,
            "val_best_epoch": metrics.val_best_epoch,
            "val_to_test_gap_map_50_95": metrics.val_map_50_95 - metrics.test_map_50_95,
            "test_ap": metrics.test_ap,
        },
        "_outcomes": outcomes,
        "_metrics": metrics,
    }


# --- validity ---------------------------------------------------------------------------------


def missing_files(results_dir: Path) -> list[str]:
    return [
        str(Path(arm.results_dir) / f)
        for arm in ARMS.values()
        for f in RUN_FILES
        if not (results_dir / arm.results_dir / f).exists()
    ]


def run_problems(arm: Arm, results_dir: Path, run1_config: dict) -> list[str]:
    """Reasons this arm's run is void by the config check (recomputed, not trusted)."""
    d = results_dir / arm.results_dir
    problems = config_problems(
        arm, run1_config, json.loads((d / "training_config.json").read_text())
    )
    record = json.loads((d / "run_record.json").read_text())
    if record.get("run") != arm.key:
        problems.append(f"run_record.json says run {record.get('run')!r}, expected {arm.key!r}")
    check = record.get("config_check") or {}
    if not check.get("ok"):
        problems.append(f"the notebook's own config check did not pass: {check.get('problems')}")
    return problems


def cross_problems(results_dir: Path) -> list[str]:
    """The two runs must share one commit, one library set and one split."""
    recs = {
        k: json.loads((results_dir / a.results_dir / "run_record.json").read_text())
        for k, a in ARMS.items()
    }
    problems = []
    for field in ("commit", "rfdetr", "albumentations", "split_sizes", "gpu", "pred_threshold"):
        if recs["2C"].get(field) != recs["2A"].get(field):
            problems.append(
                f"{field} differs: 2C {recs['2C'].get(field)!r}, 2A {recs['2A'].get(field)!r}"
            )
    return problems


# --- the whole evaluation ----------------------------------------------------------------------


def evaluate_run2(prepared_dir: Path, results_dir: Path) -> dict:
    absent = missing_files(results_dir)
    if absent:
        raise FileNotFoundError(
            "both Run 2 arms must be complete before either is evaluated; missing: "
            + ", ".join(absent)
        )
    run1_config = json.loads((results_dir / "training_config.json").read_text())
    problems = {a.key: run_problems(a, results_dir, run1_config) for a in ARMS.values()}
    cross = cross_problems(results_dir)

    gt = {
        s: load_ground_truth(prepared_dir / OUTPUT_DIRS[s] / "_annotations.coco.json")
        for s in OUTPUT_DIRS_USED
    }
    ring_b = read_ring_b(results_dir / "backdrop_boxes.csv")
    scored = {
        "run1": score_run(results_dir, gt, ring_b),
        "2C": score_run(results_dir / ARMS["2C"].results_dir, gt, ring_b),
        "2A": score_run(results_dir / ARMS["2A"].results_dir, gt, ring_b),
    }
    _check_run1_reproduces(scored["run1"], results_dir / "backdrop_boxes.csv")

    c, a = scored["2C"], scored["2A"]
    b_n, c_n = paired_counts(c["_outcomes"], a["_outcomes"])
    p = mcnemar_one_sided(b_n, c_n)
    guard = guardrails(c["_metrics"], a["_metrics"])
    m_c, k_c = c["counts"][MISREAD], c["counts"][TARGET]
    m_a, k_a = a["counts"][MISREAD], a["counts"][TARGET]
    state, criteria = decide(m_c, k_c, m_a, k_a, p, guard["all_ok"])
    invalid = [f"{k}: {x}" for k, v in problems.items() for x in v] + cross
    if invalid:
        state = "invalid"

    for s in scored.values():
        del s["_outcomes"], s["_metrics"]
    return {
        "protocol": "NOTES.md - Run 2 (colour-cast augmentation)",
        "verdict": state,
        "consequence": CONSEQUENCE[state],
        "invalid_reasons": invalid,
        "primary": {
            "M_2C": m_c,
            "K_2C": k_c,
            "M_2A": m_a,
            "K_2A": k_a,
            "mcnemar_b_misread_by_2C_only": b_n,
            "mcnemar_c_misread_by_2A_only": c_n,
            "p_one_sided": p,
            "criteria": criteria,
        },
        "guardrails": guard,
        "runs": scored,
        "limits": [
            "The test date is no longer untouched: its errors chose the question. These numbers "
            "say whether the known failure is fixed on this date, not how well the model does on "
            "a new one.",
            "One seed per arm. McNemar covers which boxes are in the test set, not training "
            "randomness; 2C against Run 1 is the only look at that.",
            "The augmentation simulates the colour of the light. A fix does not show that white "
            "balance or illumination was the real cause.",
        ],
    }


def _check_run1_reproduces(run1: dict, backdrop_csv: Path) -> None:
    """Run 1, scored through this module, must give the outcomes recorded in the backdrop check.

    If not, this evaluator differs from the one that produced Run 1's numbers, and no Run 2
    number from it can be trusted.
    """
    recorded: Counter[str] = Counter()
    with backdrop_csv.open(newline="") as f:
        for r in csv.DictReader(f):
            o = r["outcome"]
            recorded[o if o in (TARGET, MISREAD, MISSED) else OTHER] += 1
    expected = {k: recorded[k] for k in (TARGET, MISREAD, MISSED, OTHER)}
    if expected != run1["counts"]:
        raise ValueError(
            f"Run 1 does not reproduce through the Run 2 evaluator: recorded {expected}, "
            f"scored {run1['counts']}"
        )


# --- report -----------------------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{x:.3f}"


def format_report(report: dict) -> str:
    p, g, runs = report["primary"], report["guardrails"], report["runs"]
    cr = p["criteria"]
    lines = [
        f"**Verdict: {report['verdict']}** - {report['consequence']}",
        "",
    ]
    if report["invalid_reasons"]:
        lines += ["**Invalid:**"] + [f"- {r}" for r in report["invalid_reasons"]] + [""]
    lines += [
        "### Primary: the 100 test 1.5 kg boxes",
        "| run | threshold | read 1.5kg (K) | read 0.5kg (M) | missed | other |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for key, label in (("run1", "Run 1 (reference)"), ("2C", "2C control"), ("2A", "2A colour")):
        r = runs[key]
        n = r["counts"]
        lines.append(
            f"| {label} | {r['threshold']:.2f} | {n[TARGET]} | {n[MISREAD]} | {n[MISSED]} | "
            f"{n[OTHER]} |"
        )
    lines += [
        "",
        f"Paired, 2A vs 2C: b = {p['mcnemar_b_misread_by_2C_only']} misread by 2C only, "
        f"c = {p['mcnemar_c_misread_by_2A_only']} misread by 2A only; "
        f"one-sided exact McNemar p = {p['p_one_sided']:.4f}.",
        "",
        "| criterion | holds |",
        "|---|---|",
        f"| void: M(2C) <= {VOID_AT_OR_BELOW} | {cr['void']} |",
        f"| (a) M(2A) <= floor(M(2C) / 2) | {cr['a_halved']} |",
        f"| (b) p < {ALPHA} | {cr['b_significant']} |",
        f"| (c) K(2A) - K(2C) >= (M(2C) - M(2A)) / 2 | {cr['c_correct_not_missed']} |",
        f"| (d) G1, G2, G3 | {cr['d_guardrails']} |",
        "",
        "### Guardrails, 2A against 2C",
        "| | 2C | 2A | margin | holds |",
        "|---|--:|--:|--:|---|",
        f"| G1 test mAP@50-95 | {_pct(g['G1']['2C'])} | {_pct(g['G1']['2A'])} | -{MARGIN_MAP} "
        f"| {g['G1']['ok']} |",
        f"| G2 val mAP@50-95 (best) | {_pct(g['G2']['2C'])} | {_pct(g['G2']['2A'])} | "
        f"-{MARGIN_MAP} | {g['G2']['ok']} |",
    ]
    for cls, v in g["G3"]["classes"].items():
        lines.append(
            f"| G3 test AP {cls} | {_pct(v['2C'])} | {_pct(v['2A'])} | -{MARGIN_CLASS_AP} "
            f"| {v['ok']} |"
        )
    lines += [
        "",
        "### Secondary (decides nothing)",
        "| run | M yellow | K yellow | M blue | K blue | 0.5kg precision | test mAP@50 | "
        "test mAP@50-95 | val mAP@50-95 | val to test gap |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for key, label in (("run1", "Run 1"), ("2C", "2C"), ("2A", "2A")):
        r, m = runs[key], runs[key]["metrics"]
        y, b = r["by_side"]["yellow"], r["by_side"]["blue"]
        lines.append(
            f"| {label} | {y[MISREAD]} | {y[TARGET]} | {b[MISREAD]} | {b[TARGET]} | "
            f"{_pct(r['precision_0.5kg'])} | {_pct(m['test_map_50'])} | "
            f"{_pct(m['test_map_50_95'])} | {_pct(m['val_map_50_95'])} | "
            f"{_pct(m['val_to_test_gap_map_50_95'])} |"
        )
    lines += ["", "Full test confusion tables are in the JSON (`runs.<run>.confusion`).", ""]
    lines += ["### What this cannot show"] + [f"- {t}" for t in report["limits"]]
    return "\n".join(lines)
