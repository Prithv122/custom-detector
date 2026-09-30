"""Baseline gate and five-arm run, with a fake ``predict_fn`` standing in for the model.

No real inference here — these tests only check the wiring (gate matching, arm box
selection, the decision rule, the report). The one real-model run is a separate, one-off
script, not something CI can reproduce without the checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from detector.evaluate import Detection, load_ground_truth
from detector.recolour import KEEP, REVERSE, RING, SHAM, WHOLE
from detector.recolour_run import (
    BackdropRow,
    GateCheck,
    best_iou_confidence,
    decide,
    is_success,
    name_to_file,
    read_backdrop_rows,
    run_arm,
    run_baseline_gate,
    select_arm_boxes,
    summarise,
)

CATS = [{"id": 1, "name": "0.5kg"}, {"id": 2, "name": "1.5kg"}]
PLATE = (40.0, 40.0, 16.0, 60.0)


def _write_split(root: Path, images: list[dict]) -> Path:
    root.mkdir(parents=True)
    coco = {"categories": CATS, "images": [], "annotations": []}
    for i, spec in enumerate(images):
        w, h = spec.get("size", (120, 160))
        arr = np.full((h, w, 3), spec.get("bg", (128, 128, 128)), dtype=np.uint8)
        for cat, box in spec["boxes"]:
            coco["annotations"].append(
                {
                    "id": len(coco["annotations"]),
                    "image_id": i,
                    "category_id": cat,
                    "bbox": list(box),
                }
            )
        fname = f"{spec['name'].replace('.', '_')}.rf.{i:032x}.png"
        Image.fromarray(arr).save(root / fname)
        coco["images"].append(
            {"id": i, "file_name": fname, "width": w, "height": h, "extra": {"name": spec["name"]}}
        )
    (root / "_annotations.coco.json").write_text(json.dumps(coco))
    return root


def _write_backdrop_csv(path: Path, rows: list[tuple]) -> Path:
    lines = ["image,outcome,ring_pixels,ring_b,plate_chroma,group"]
    lines += [",".join(str(v) for v in r) for r in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_name_to_file_prefers_the_original_name() -> None:
    coco = {"images": [{"file_name": "x.rf.abc.png", "extra": {"name": "orig.jpg"}}]}
    assert name_to_file(coco) == {"orig.jpg": "x.rf.abc.png"}


def test_is_success_wants_1_5kg_except_for_reverse() -> None:
    assert is_success(RING, "1.5kg") and not is_success(RING, "0.5kg")
    assert is_success(SHAM, "1.5kg")
    assert is_success(WHOLE, "1.5kg")
    assert is_success(KEEP, "1.5kg") and not is_success(KEEP, "0.5kg")
    assert is_success(REVERSE, "0.5kg") and not is_success(REVERSE, "1.5kg")


def test_best_iou_confidence_ignores_wrong_class_and_low_iou() -> None:
    box = (0.0, 0.0, 10.0, 10.0)
    preds = [
        Detection("1.5kg", 0.3, (0.0, 0.0, 10.0, 10.0)),  # good match, low conf
        Detection("1.5kg", 0.9, (50.0, 50.0, 10.0, 10.0)),  # high conf, no overlap
        Detection("0.5kg", 0.99, (0.0, 0.0, 10.0, 10.0)),  # perfect overlap, wrong class
    ]
    assert best_iou_confidence(preds, box, "1.5kg") == pytest.approx(0.3)
    assert best_iou_confidence([], box, "1.5kg") == 0.0


@pytest.mark.parametrize(
    ("n", "ring", "whole", "sham", "expected"),
    [
        (40, 25, 3, 0, "for (local)"),
        (40, 10, 22, 0, "for (scene)"),
        (40, 2, 1, 0, "against"),
        (40, 15, 12, 0, "inconclusive"),
        (40, 30, 5, 5, "pipeline noise"),  # sham noise wins even though ring clears the bar
    ],
)
def test_decision_rule(n, ring, whole, sham, expected) -> None:
    assert decide(n, ring, whole, sham) == expected


def test_read_backdrop_rows_parses_missing_ring_b_as_none(tmp_path: Path) -> None:
    csv_path = _write_backdrop_csv(
        tmp_path / "b.csv",
        [
            ("a.jpg", "0.5kg", 500, 40.0, 30.0, "misread"),
            ("b.jpg", "missed", 0, "", "", "missed"),
        ],
    )
    rows = read_backdrop_rows(csv_path)
    assert rows[0] == BackdropRow("a.jpg", "0.5kg", "misread", 40.0)
    assert rows[1].ring_b is None


def test_baseline_gate_flags_mismatches_and_skips_non_arm_rows(tmp_path: Path) -> None:
    split = _write_split(
        tmp_path / "test",
        [
            {"name": "a.jpg", "boxes": [(2, PLATE)]},  # recorded misread, local agrees
            {"name": "b.jpg", "boxes": [(2, PLATE)]},  # recorded correct, local disagrees
            {"name": "c.jpg", "boxes": [(2, PLATE)]},  # missed row: not checked at all
        ],
    )
    coco = json.loads((split / "_annotations.coco.json").read_text())
    gt = load_ground_truth(split / "_annotations.coco.json")
    rows = [
        BackdropRow("a.jpg", "0.5kg", "misread", 40.0),
        BackdropRow("b.jpg", "1.5kg", "correct", -10.0),
        BackdropRow("c.jpg", "missed", "missed", None),
    ]

    def predict_fn(rgb: np.ndarray) -> list[Detection]:
        return [Detection("0.5kg", 0.9, PLATE)]  # always reads 0.5kg locally

    checks = run_baseline_gate(predict_fn, split, coco, gt, rows)

    assert {c.image for c in checks} == {"a.jpg", "b.jpg"}  # c.jpg (missed) skipped
    by_name = {c.image: c for c in checks}
    assert by_name["a.jpg"].ok
    assert not by_name["b.jpg"].ok


def test_select_arm_boxes_splits_correct_by_ring_b_sign_and_drops_gate_failures() -> None:
    rows = [
        BackdropRow("m1.jpg", "0.5kg", "misread", 40.0),
        BackdropRow("m2.jpg", "0.5kg", "misread", 30.0),  # fails gate, dropped everywhere
        BackdropRow("cb.jpg", "1.5kg", "correct", -5.0),
        BackdropRow("cy.jpg", "1.5kg", "correct", 5.0),
        BackdropRow("cy2.jpg", "1.5kg", "correct", None),  # unmeasurable, excluded
    ]
    gate = [
        GateCheck("m1.jpg", "misread", "0.5kg", "0.5kg"),
        GateCheck("m2.jpg", "misread", "0.5kg", "1.5kg"),  # mismatch
        GateCheck("cb.jpg", "correct", "1.5kg", "1.5kg"),
        GateCheck("cy.jpg", "correct", "1.5kg", "1.5kg"),
        GateCheck("cy2.jpg", "correct", "1.5kg", "1.5kg"),
    ]

    arms = select_arm_boxes(rows, gate)

    assert arms[RING] == arms[SHAM] == arms[WHOLE] == ["m1.jpg"]
    assert arms[REVERSE] == ["cb.jpg"]
    assert arms[KEEP] == ["cy.jpg"]


def test_run_arm_records_flip_and_confidence_change(tmp_path: Path) -> None:
    split = _write_split(
        tmp_path / "test", [{"name": "a.jpg", "bg": (230, 220, 40), "boxes": [(2, PLATE)]}]
    )
    coco = json.loads((split / "_annotations.coco.json").read_text())
    gt = load_ground_truth(split / "_annotations.coco.json")
    targets = {"blue": (0.0, -40.0), "yellow": (0.0, 40.0)}
    baseline_preds = {"a.jpg": [Detection("1.5kg", 0.2, PLATE)]}

    def predict_fn(rgb: np.ndarray) -> list[Detection]:
        return [Detection("1.5kg", 0.95, PLATE)]  # recolouring "fixed" the read

    [outcome] = run_arm(predict_fn, split, coco, gt, targets, RING, ["a.jpg"], baseline_preds)

    assert outcome.new_outcome == "1.5kg"
    assert outcome.success is True
    assert outcome.baseline_target_confidence == pytest.approx(0.2)
    assert outcome.new_target_confidence == pytest.approx(0.95)


def test_summarise_reports_gate_failure_without_a_verdict() -> None:
    gate = [
        GateCheck("a.jpg", "misread", "0.5kg", "1.5kg"),
        GateCheck("b.jpg", "misread", "0.5kg", "1.5kg"),
        GateCheck("c.jpg", "misread", "0.5kg", "1.5kg"),
    ]
    s = summarise(gate, {}, {"blue": None, "yellow": None}, "deadbeef")
    assert s["baseline_gate"]["n_dropped"] == 3
    assert s["baseline_gate"]["passed"] is False
    assert s["verdict"] == "no verdict — baseline gate failed"
