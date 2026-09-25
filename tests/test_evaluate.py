import json
from collections import Counter
from pathlib import Path

import pytest

from detector.evaluate import (
    BACKGROUND,
    MISSED,
    Detection,
    ap50,
    error_breakdown,
    evaluate,
    iou,
    match_image,
    micro_f1,
    pick_threshold,
    verdict,
    wilson,
)

A = (0.0, 0.0, 10.0, 10.0)
B = (100.0, 0.0, 10.0, 10.0)


def test_iou_of_identical_disjoint_and_half_overlapping_boxes() -> None:
    assert iou(A, A) == 1.0
    assert iou(A, B) == 0.0
    assert iou(A, (5.0, 0.0, 10.0, 10.0)) == pytest.approx(50 / 150)


def test_matching_ignores_class_so_a_confusion_is_visible() -> None:
    pairs = match_image([("0.5kg", A)], [Detection("5kg", 0.9, A)], threshold=0.5)
    assert pairs == [("0.5kg", "5kg")]


def test_higher_confidence_prediction_claims_the_box_even_if_wrong() -> None:
    gt = [("1.5kg", A)]
    preds = [Detection("1.5kg", 0.6, A), Detection("15kg", 0.8, A)]
    assert match_image(gt, preds, threshold=0.5) == [("1.5kg", "15kg"), (BACKGROUND, "1.5kg")]


def test_low_iou_low_confidence_and_leftovers() -> None:
    gt = [("25kg", A), ("2kg", B)]
    preds = [
        Detection("25kg", 0.9, (6.0, 0.0, 10.0, 10.0)),  # IoU 0.25 → background
        Detection("2kg", 0.3, B),  # below threshold → dropped
    ]
    pairs = match_image(gt, preds, threshold=0.5)
    assert sorted(pairs) == sorted([(BACKGROUND, "25kg"), ("25kg", MISSED), ("2kg", MISSED)])


def test_each_ground_truth_box_is_matched_at_most_once() -> None:
    preds = [Detection("25kg", 0.9, A), Detection("25kg", 0.8, A)]
    assert match_image([("25kg", A)], preds, 0.5) == [("25kg", "25kg"), (BACKGROUND, "25kg")]


def test_error_breakdown_rows_sum_to_ground_truth() -> None:
    table = Counter(
        {
            ("0.5kg", "0.5kg"): 5,
            ("0.5kg", "5kg"): 3,
            ("0.5kg", "1kg"): 1,
            ("0.5kg", MISSED): 1,
            ("zacisk", "25kg"): 2,
        }
    )
    rows = error_breakdown(table)
    r = rows["0.5kg"]
    assert (r["n_gt"], r["correct"], r["partner"], r["other"], r["missed"]) == (10, 5, 3, 1, 1)
    assert r["partner_class"] == "5kg" and r["partner_share"] == 0.3
    # the collar has no colour partner: everything wrong is "other"
    z = rows["zacisk"]
    assert z["partner_class"] is None and z["other"] == 2


def test_micro_f1_counts_confusions_as_both_a_false_positive_and_a_miss() -> None:
    table = Counter({("1kg", "1kg"): 2, ("1kg", "10kg"): 1, (BACKGROUND, "2kg"): 1})
    # correct 2, predictions 4, ground truth 3
    assert micro_f1(table) == pytest.approx(4 / 7)


def test_threshold_is_chosen_to_drop_low_confidence_noise() -> None:
    gt = {"a.jpg": [("25kg", A)]}
    preds = {"a.jpg": [Detection("25kg", 0.9, A), Detection("25kg", 0.2, B)]}
    t, f1 = pick_threshold(gt, preds)
    assert 0.2 < t <= 0.9 and f1 == 1.0


def test_wilson_interval_contains_the_estimate_and_stays_in_range() -> None:
    lo, hi = wilson(3, 10)
    assert 0 <= lo < 0.3 < hi <= 1
    assert wilson(0, 0) == (0.0, 0.0)


def test_ap50_is_one_for_a_perfect_ranking_and_lower_when_a_miss_ranks_first() -> None:
    gt = {"a.jpg": [("25kg", A)], "b.jpg": [("25kg", A)]}
    perfect = {"a.jpg": [Detection("25kg", 0.9, A)], "b.jpg": [Detection("25kg", 0.8, A)]}
    assert ap50(gt, perfect)["25kg"] == pytest.approx(1.0)
    noisy = {**perfect, "a.jpg": [Detection("25kg", 0.95, B), Detection("25kg", 0.9, A)]}
    assert ap50(gt, noisy)["25kg"] < 1.0


def _rows(**shares: tuple[int, int, int, int]) -> dict[str, dict]:
    """class -> (partner, other, missed, n)."""
    out = {}
    for cls in ("0.5kg", "1.5kg", "1kg", "2kg", "2.5kg"):
        partner, other, missed, n = shares.get(cls, (0, 0, 0, 100))
        out[cls] = {
            "partner": partner,
            "other": other,
            "missed": missed,
            "partner_share": partner / n,
        }
    return out


def test_verdict_supported_only_when_all_three_conditions_hold_for_both_classes() -> None:
    val = _rows(**{"0.5kg": (2, 0, 1, 100), "1.5kg": (1, 0, 1, 100)})
    test = _rows(**{"0.5kg": (20, 3, 5, 100), "1.5kg": (15, 2, 4, 100), "1kg": (3, 0, 2, 100)})
    assert verdict(val, test)[0] == "supported"
    # same test numbers, but 1.5 kg was already this confused on val → condition (c) fails
    val_high = _rows(**{"0.5kg": (2, 0, 1, 100), "1.5kg": (20, 0, 1, 100)})
    assert verdict(val_high, test)[0] == "mixed"


def test_verdict_rejected_when_both_classes_are_mostly_missed() -> None:
    val = _rows()
    test = _rows(**{"0.5kg": (3, 1, 25, 100), "1.5kg": (2, 1, 20, 100)})
    assert verdict(val, test)[0] == "rejected"


def test_evaluate_end_to_end_on_tiny_files(tmp_path: Path) -> None:
    cats = [{"id": 0, "name": "objects"}, {"id": 1, "name": "0.5kg"}, {"id": 2, "name": "5kg"}]
    for folder in ("valid", "test"):
        d = tmp_path / "prep" / folder
        d.mkdir(parents=True)
        coco = {
            "categories": cats,
            "images": [{"id": 0, "file_name": "x.rf.jpg", "extra": {"name": "x.jpg"}}],
            "annotations": [{"id": 0, "image_id": 0, "category_id": 1, "bbox": list(A)}],
        }
        (d / "_annotations.coco.json").write_text(json.dumps(coco))
    res = tmp_path / "results"
    res.mkdir()
    header = "file,class_name,confidence,x,y,w,h\n"
    (res / "predictions_val.csv").write_text(header + "x.jpg,0.5kg,0.9,0,0,10,10\n")
    (res / "predictions_test.csv").write_text(header + "x.jpg,5kg,0.9,0,0,10,10\n")

    report = evaluate(tmp_path / "prep", res)

    assert report["val"]["errors"]["0.5kg"]["correct"] == 1
    assert report["test"]["errors"]["0.5kg"]["partner"] == 1
    assert report["test"]["per_class"]["5kg"]["precision"] == 0.0
