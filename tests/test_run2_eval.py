"""Run 2 evaluator: the decision rule, McNemar, guardrails, and each verdict on a synthetic world.

Everything here runs on made-up predictions. The one test on real files scores Run 1 through the
Run 2 code and must reproduce the outcomes recorded before Run 2 existed.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import pytest

from detector.cli import build_parser
from detector.dataset import CLASSES
from detector.evaluate import Detection
from detector.run2 import ARMS
from detector.run2_eval import (
    CONSEQUENCE,
    STATES,
    Metrics,
    decide,
    evaluate_run2,
    format_report,
    guardrails,
    mcnemar_one_sided,
    paired_counts,
    read_metrics,
    score_run,
    target_outcomes,
)

ROOT = Path(__file__).resolve().parents[1]
RUN1_CONFIG = ROOT / "results" / "training_config.json"
BOX = (10.0, 10.0, 20.0, 60.0)
N = 100


# --- McNemar ----------------------------------------------------------------------------------


def test_mcnemar_is_the_exact_one_sided_binomial() -> None:
    assert mcnemar_one_sided(0, 0) == 1.0
    assert mcnemar_one_sided(10, 0) == pytest.approx(2**-10)
    assert mcnemar_one_sided(8, 2) == pytest.approx(sum(math.comb(10, k) for k in range(3)) / 1024)
    assert mcnemar_one_sided(5, 5) > 0.5  # a tie is not evidence
    assert mcnemar_one_sided(2, 8) > 0.95  # 2A misreading more is not evidence either


def test_mcnemar_p_falls_as_the_treatment_wins_more_discordant_boxes() -> None:
    ps = [mcnemar_one_sided(b, 2) for b in range(2, 20)]
    assert ps == sorted(ps, reverse=True)


def test_paired_counts_need_the_same_boxes() -> None:
    c = {"a": "0.5kg", "b": "0.5kg", "c": "1.5kg", "d": "missed"}
    a = {"a": "1.5kg", "b": "0.5kg", "c": "0.5kg", "d": "missed"}
    assert paired_counts(c, a) == (1, 1)
    with pytest.raises(ValueError, match="same ground-truth boxes"):
        paired_counts(c, {"a": "1.5kg"})


# --- the decision rule ------------------------------------------------------------------------


def test_every_state_is_reachable_and_described() -> None:
    assert set(CONSEQUENCE) == set(STATES) | {"invalid"}


@pytest.mark.parametrize(
    ("m_c", "k_c", "m_a", "k_a", "p", "guard", "state"),
    [
        (40, 36, 10, 66, 1e-6, True, "fixed"),
        (40, 36, 10, 66, 1e-6, False, "fixed at a cost"),
        (40, 36, 10, 36, 1e-6, True, "displaced"),
        (40, 36, 10, 36, 1e-6, False, "displaced"),  # (c) failing decides it, whatever G says
        (40, 36, 40, 36, 1.0, True, "not fixed"),
        (40, 36, 25, 51, 1e-4, True, "not fixed"),  # 25 > floor(40 / 2)
        (40, 36, 10, 66, 0.2, True, "not fixed"),  # halved but not significant
        (20, 60, 0, 80, 1e-6, True, "void"),  # control already at the void line
        (15, 60, 15, 60, 1.0, True, "void"),
    ],
)
def test_decision_rule(
    m_c: int, k_c: int, m_a: int, k_a: int, p: float, guard: bool, state: str
) -> None:
    assert decide(m_c, k_c, m_a, k_a, p, guard)[0] == state


def test_void_line_is_inclusive_at_20_and_not_at_21() -> None:
    assert decide(20, 60, 5, 75, 1e-6, True)[0] == "void"
    assert decide(21, 60, 5, 76, 1e-6, True)[0] == "fixed"


def test_halving_uses_the_floor() -> None:
    assert decide(41, 36, 20, 57, 1e-6, True)[1]["a_halved"] is True  # 20 <= floor(41/2)
    assert decide(41, 36, 21, 56, 1e-6, True)[1]["a_halved"] is False


def test_misreads_must_mostly_become_correct_reads_not_misses() -> None:
    # 30 fewer misreads: at least 15 of them must be new correct reads.
    assert (
        decide(40, 36, 10, 51, 1e-6, True)[1]["c_correct_not_missed"] is True
    )  # +15, exactly half
    assert decide(40, 36, 10, 50, 1e-6, True)[1]["c_correct_not_missed"] is False  # +14


# --- guardrails -------------------------------------------------------------------------------


def _metrics(test: float = 0.74, val: float = 0.83, ap: float = 0.7) -> Metrics:
    return Metrics(test, 0.95, val, 36, dict.fromkeys(CLASSES, ap))


def test_guardrail_margins_hold_at_the_boundary_and_fail_beyond_it() -> None:
    c = _metrics()
    assert guardrails(c, _metrics(test=0.72))["G1"]["ok"]
    assert not guardrails(c, _metrics(test=0.7199))["G1"]["ok"]
    assert guardrails(c, _metrics(val=0.81))["G2"]["ok"]
    assert not guardrails(c, _metrics(val=0.8099))["G2"]["ok"]
    assert guardrails(c, _metrics(ap=0.65))["G3"]["ok"]
    assert not guardrails(c, _metrics(ap=0.6499))["G3"]["ok"]


def test_g3_skips_the_two_hypothesis_classes_and_names_the_culprit() -> None:
    c = _metrics()
    t = Metrics(0.74, 0.95, 0.83, 36, {**dict.fromkeys(CLASSES, 0.7), "0.5kg": 0.1, "5kg": 0.6})
    g = guardrails(c, t)
    assert "0.5kg" not in g["G3"]["classes"] and "1.5kg" not in g["G3"]["classes"]
    assert [k for k, v in g["G3"]["classes"].items() if not v["ok"]] == ["5kg"]
    assert not g["all_ok"]


def test_a_gain_is_never_a_failure() -> None:
    assert guardrails(_metrics(), _metrics(test=0.9, val=0.9, ap=0.9))["all_ok"]


# --- reading metrics --------------------------------------------------------------------------


def _write_metrics(path: Path, val: list[tuple[float, float]], test: float = 0.74) -> None:
    cols = ["epoch", "val/mAP_50_95", "val/ema_mAP_50_95", "test/mAP_50_95", "test/mAP_50"]
    cols += [f"test/AP/{c}" for c in CLASSES]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for e, (v, ema) in enumerate(val):
            w.writerow([e, v, ema, "", "", *[""] * len(CLASSES)])
        w.writerow([len(val) - 1, "", "", test, 0.95, *[0.7] * len(CLASSES)])


def test_read_metrics_takes_the_best_val_epoch_and_the_single_test_row(tmp_path: Path) -> None:
    _write_metrics(tmp_path / "m.csv", [(0.5, 0.5), (0.8, 0.82), (0.7, 0.7)])
    m = read_metrics(tmp_path / "m.csv")
    assert (m.val_map_50_95, m.val_best_epoch, m.test_map_50_95) == (0.82, 1, 0.74)
    assert set(m.test_ap) == set(CLASSES)


def test_read_metrics_refuses_a_file_with_two_test_rows(tmp_path: Path) -> None:
    _write_metrics(tmp_path / "m.csv", [(0.5, 0.5)])
    with (tmp_path / "m.csv").open("a", newline="") as f:
        csv.writer(f).writerow([1, "", "", 0.7, 0.9, *[0.7] * len(CLASSES)])
    with pytest.raises(ValueError, match="one test row"):
        read_metrics(tmp_path / "m.csv")


# --- outcome of the 1.5 kg box ----------------------------------------------------------------


def test_target_outcomes_name_all_four_fates() -> None:
    gt = {
        "a": [("1.5kg", BOX)],
        "b": [("1.5kg", BOX)],
        "c": [("1.5kg", BOX)],
        "d": [("1.5kg", BOX)],
        "e": [("5kg", BOX)],  # no 1.5 kg box: not counted
    }
    preds = {
        "a": [Detection("1.5kg", 0.9, BOX)],
        "b": [Detection("0.5kg", 0.9, BOX)],
        "c": [Detection("15kg", 0.9, BOX)],
        "d": [Detection("1.5kg", 0.1, BOX)],  # under the threshold
    }
    assert target_outcomes(gt, preds, 0.5) == {
        "a": "1.5kg",
        "b": "0.5kg",
        "c": "other",
        "d": "missed",
    }


def test_target_outcomes_refuse_two_boxes_in_one_image() -> None:
    with pytest.raises(ValueError, match="expected 1"):
        target_outcomes({"a": [("1.5kg", BOX), ("1.5kg", (80.0, 10.0, 20.0, 60.0))]}, {}, 0.5)


# --- a synthetic world, end to end ------------------------------------------------------------


def pattern(m: int, k: int, missed: int, other: int) -> list[str]:
    out = ["0.5kg"] * m + ["1.5kg"] * k + ["missed"] * missed + ["other"] * other
    assert len(out) == N
    return out


BASE = pattern(40, 36, 21, 3)  # Run 1's outcomes
CLASS_FOR = {"1.5kg": "1.5kg", "0.5kg": "0.5kg", "other": "1kg"}


def _fixed(base: list[str]) -> list[str]:
    return ["1.5kg" if 10 <= i < 40 else o for i, o in enumerate(base)]


def _displaced(base: list[str]) -> list[str]:
    return ["missed" if 10 <= i < 40 else o for i, o in enumerate(base)]


def _write_run(
    d: Path, outcomes: list[str], *, test_map: float = 0.74, config: dict | None = None
) -> None:
    d.mkdir(parents=True, exist_ok=True)
    with (d / "predictions_test.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "class_name", "confidence", "x", "y", "w", "h"])
        for i, o in enumerate(outcomes):
            if o != "missed":
                w.writerow([f"img{i:03d}.jpg", CLASS_FOR[o], "0.9000", *BOX])
    with (d / "predictions_val.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "class_name", "confidence", "x", "y", "w", "h"])
        for i in range(4):
            w.writerow([f"val{i}.jpg", "1.5kg", "0.9000", *BOX])
    _write_metrics(d / "metrics.csv", [(0.8, 0.8), (0.83, 0.83)], test=test_map)
    if config is not None:
        (d / "training_config.json").write_text(json.dumps(config))


def build_world(
    tmp: Path,
    c: list[str],
    a: list[str],
    *,
    a_test_map: float = 0.74,
    run1: list[str] | None = None,
) -> tuple[Path, Path]:
    prepared, results = tmp / "prepared", tmp / "results"
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(CLASSES)]
    cid = {n: i + 1 for i, n in enumerate(CLASSES)}
    for split, n_img, prefix in (("valid", 4, "val"), ("test", N, "img")):
        (prepared / split).mkdir(parents=True)
        images = [
            {"id": i, "file_name": f"x{i}.jpg", "extra": {"name": f"{prefix}{i:03d}.jpg"}}
            for i in range(n_img)
        ]
        if prefix == "val":
            images = [{**im, "extra": {"name": f"val{i}.jpg"}} for i, im in enumerate(images)]
        anns = [
            {"id": i, "image_id": i, "category_id": cid["1.5kg"], "bbox": list(BOX)}
            for i in range(n_img)
        ]
        coco = {"categories": cats, "images": images, "annotations": anns}
        (prepared / split / "_annotations.coco.json").write_text(json.dumps(coco))

    run1_config = json.loads(RUN1_CONFIG.read_text())
    results.mkdir()
    (results / "training_config.json").write_text(json.dumps(run1_config))
    _write_run(results, run1 or BASE)
    with (results / "backdrop_boxes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "outcome", "ring_b"])
        for i, o in enumerate(run1 or BASE):
            w.writerow([f"img{i:03d}.jpg", "1kg" if o == "other" else o, 5.0 if i < 76 else -5.0])
    for arm, outcomes, tmap in ((ARMS["2C"], c, 0.74), (ARMS["2A"], a, a_test_map)):
        saved = json.loads(json.dumps(run1_config))
        saved["train_config"].update(json.loads(json.dumps(arm.train_kwargs())))
        _write_run(results / arm.results_dir, outcomes, test_map=tmap, config=saved)
        record = {
            "run": arm.key,
            "commit": "abc123",
            "rfdetr": "1.11.0",
            "albumentations": "2.0.8",
            "split_sizes": {"train": 1263, "val": 331, "test": 402},
            "gpu": "Tesla T4",
            "pred_threshold": 0.01,
            "config_check": {"ok": True, "problems": []},
        }
        (results / arm.results_dir / "run_record.json").write_text(json.dumps(record))
    return prepared, results


@pytest.mark.parametrize(
    ("make_a", "a_test_map", "state"),
    [
        (_fixed, 0.74, "fixed"),
        (_fixed, 0.70, "fixed at a cost"),  # G1 fails: test mAP down 0.04
        (_displaced, 0.74, "displaced"),
        (lambda base: list(base), 0.74, "not fixed"),
    ],
)
def test_each_verdict_on_a_synthetic_world(
    tmp_path: Path, make_a, a_test_map: float, state: str
) -> None:
    prepared, results = build_world(tmp_path, BASE, make_a(BASE), a_test_map=a_test_map)
    report = evaluate_run2(prepared, results)
    assert report["verdict"] == state
    assert report["invalid_reasons"] == []
    assert report["runs"]["run1"]["counts"] == {"1.5kg": 36, "0.5kg": 40, "missed": 21, "other": 3}


def test_the_fixed_world_reports_the_numbers_the_protocol_asks_for(tmp_path: Path) -> None:
    prepared, results = build_world(tmp_path, BASE, _fixed(BASE))
    report = evaluate_run2(prepared, results)
    p = report["primary"]
    assert (p["M_2C"], p["K_2C"], p["M_2A"], p["K_2A"]) == (40, 36, 10, 66)
    assert (p["mcnemar_b_misread_by_2C_only"], p["mcnemar_c_misread_by_2A_only"]) == (30, 0)
    assert p["p_one_sided"] == pytest.approx(2**-30)
    assert report["runs"]["2C"]["by_side"]["yellow"]["0.5kg"] == 40  # all on the yellow side
    assert report["runs"]["2A"]["by_side"]["yellow"]["0.5kg"] == 10
    assert report["runs"]["2A"]["metrics"]["val_to_test_gap_map_50_95"] == pytest.approx(0.09)
    assert "Verdict: fixed" in format_report(report)


def test_a_control_that_misreads_little_is_void(tmp_path: Path) -> None:
    c = pattern(15, 61, 21, 3)
    prepared, results = build_world(tmp_path, c, list(c))
    assert evaluate_run2(prepared, results)["verdict"] == "void"


def test_a_run_that_fails_the_config_check_has_no_verdict(tmp_path: Path) -> None:
    prepared, results = build_world(tmp_path, BASE, _fixed(BASE))
    path = results / "run2a" / "training_config.json"
    saved = json.loads(path.read_text())
    saved["train_config"]["lr"] = 2e-4
    path.write_text(json.dumps(saved))
    report = evaluate_run2(prepared, results)
    assert report["verdict"] == "invalid"
    assert any("train_config.lr" in r for r in report["invalid_reasons"])
    assert report["primary"]["M_2A"] == 10  # the numbers are still reported


def test_runs_on_different_commits_are_not_comparable(tmp_path: Path) -> None:
    prepared, results = build_world(tmp_path, BASE, _fixed(BASE))
    path = results / "run2a" / "run_record.json"
    record = json.loads(path.read_text())
    record["commit"] = "def456"
    path.write_text(json.dumps(record))
    report = evaluate_run2(prepared, results)
    assert report["verdict"] == "invalid"
    assert any(r.startswith("commit differs") for r in report["invalid_reasons"])


def test_the_notebooks_own_failed_check_is_enough_to_void(tmp_path: Path) -> None:
    prepared, results = build_world(tmp_path, BASE, _fixed(BASE))
    path = results / "run2c" / "run_record.json"
    record = json.loads(path.read_text())
    record["config_check"] = {"ok": False, "problems": ["x"]}
    path.write_text(json.dumps(record))
    assert evaluate_run2(prepared, results)["verdict"] == "invalid"


def test_nothing_is_scored_until_both_arms_are_complete(tmp_path: Path) -> None:
    prepared, results = build_world(tmp_path, BASE, _fixed(BASE))
    shutil.rmtree(results / "run2a")
    with pytest.raises(FileNotFoundError, match="run2a"):
        evaluate_run2(prepared, results)


def test_an_evaluator_that_cannot_reproduce_run_1_refuses_to_score(tmp_path: Path) -> None:
    prepared, results = build_world(tmp_path, BASE, _fixed(BASE))
    lines = (results / "backdrop_boxes.csv").read_text().splitlines()
    lines[1] = lines[1].replace("0.5kg", "1.5kg")  # one recorded outcome no longer matches
    (results / "backdrop_boxes.csv").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="does not reproduce"):
        evaluate_run2(prepared, results)


def test_a_test_set_of_the_wrong_size_is_refused(tmp_path: Path) -> None:
    prepared, results = build_world(tmp_path, BASE, _fixed(BASE))
    coco_path = prepared / "test" / "_annotations.coco.json"
    coco = json.loads(coco_path.read_text())
    coco["annotations"] = coco["annotations"][:-1]
    coco_path.write_text(json.dumps(coco))
    with pytest.raises(ValueError, match="protocol fixes 100"):
        evaluate_run2(prepared, results)


def test_cli_writes_json_and_markdown(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    prepared, results = build_world(tmp_path, BASE, _fixed(BASE))
    args = build_parser().parse_args(
        ["run2-eval", "--prepared", str(prepared), "--results", str(results)]
    )
    assert args.func(args) == 0
    assert json.loads((results / "run2_evaluation.json").read_text())["verdict"] == "fixed"
    assert "Verdict: fixed" in (results / "run2_evaluation.md").read_text(encoding="utf-8")
    assert "Verdict: fixed" in capsys.readouterr().out


# --- Run 1 through this code, on the real files -----------------------------------------------

PREPARED = ROOT / "data" / "processed" / "plates_v10_clean"


@pytest.mark.skipif(not PREPARED.exists(), reason="prepared dataset not built (git-ignored)")
def test_run_1_reproduces_through_the_run2_evaluator() -> None:
    from detector.dataset.prepare import OUTPUT_DIRS
    from detector.evaluate import load_ground_truth
    from detector.run2_eval import read_ring_b

    gt = {
        s: load_ground_truth(PREPARED / OUTPUT_DIRS[s] / "_annotations.coco.json")
        for s in ("val", "test")
    }
    r = score_run(ROOT / "results", gt, read_ring_b(ROOT / "results" / "backdrop_boxes.csv"))
    assert r["threshold"] == 0.8
    assert r["counts"] == {"1.5kg": 36, "0.5kg": 40, "missed": 21, "other": 3}
    assert r["by_side"]["yellow"]["0.5kg"] == 40 and r["by_side"]["blue"]["0.5kg"] == 0
    assert r["precision_0.5kg"] == pytest.approx(0.715, abs=5e-4)
    assert r["metrics"]["test_map_50_95"] == pytest.approx(0.741, abs=5e-4)
