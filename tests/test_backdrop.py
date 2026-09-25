"""Backdrop check on synthetic images whose answer is known before measuring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from detector.backdrop import (
    MIN_RING_PIXELS,
    BoxMeasure,
    C,
    M,
    auc,
    cluster_bootstrap_auc,
    decide,
    measure_split,
    ring_mask,
    srgb_to_lab,
    summarise,
)
from detector.evaluate import MISSED

YELLOW, BLUE, GREY = (230, 220, 40), (20, 40, 150), (128, 128, 128)
CATS = [{"id": 1, "name": "0.5kg"}, {"id": 2, "name": "1.5kg"}, {"id": 3, "name": "20kg"}]


@pytest.mark.parametrize(
    ("rgb", "lab"),
    [
        ((255, 255, 255), (100.0, 0.0, 0.0)),
        ((0, 0, 0), (0.0, 0.0, 0.0)),
        ((255, 255, 0), (97.14, -21.55, 94.48)),
        ((0, 0, 255), (32.30, 79.19, -107.86)),
    ],
)
def test_lab_matches_published_reference_values(rgb, lab) -> None:
    got = srgb_to_lab(np.array(rgb, dtype=np.uint8))
    assert got == pytest.approx(lab, abs=0.02)


def test_ring_leaves_out_every_labelled_box_and_stays_in_the_image() -> None:
    box = (10.0, 10.0, 8.0, 40.0)  # margin = max(8, 0.25 * 40) = 10
    neighbour = (18.0, 10.0, 8.0, 40.0)
    mask = ring_mask(box, [box, neighbour], (100, 100))
    assert not mask[10:50, 10:26].any()  # the plate and its neighbour
    assert mask[0:60, 0:10].all()  # left band, clipped at x = 0 and y = 0
    assert mask[50:60, 10:28].all()  # below both plates
    assert not mask[:, 28:].any() and not mask[60:, :].any()  # outside the grown box


def _write_split(root: Path, images: list[dict]) -> Path:
    """images: [{name, bg, boxes: [(cat, bbox, colour)], size?}]"""
    root.mkdir(parents=True)
    coco = {"categories": CATS, "images": [], "annotations": []}
    for i, spec in enumerate(images):
        w, h = spec.get("size", (120, 160))
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[:] = spec["bg"]
        for cat, (x, y, bw, bh), colour in spec["boxes"]:
            arr[int(y) : int(y + bh), int(x) : int(x + bw)] = colour
            coco["annotations"].append(
                {
                    "id": len(coco["annotations"]),
                    "image_id": i,
                    "category_id": cat,
                    "bbox": [x, y, bw, bh],
                }
            )
        fname = f"{spec['name'].replace('.', '_')}.rf.{i:032x}.png"
        Image.fromarray(arr).save(root / fname)
        coco["images"].append(
            {
                "id": i,
                "file_name": fname,
                "width": spec.get("coco_w", w),
                "height": h,
                "extra": {"name": spec["name"]},
            }
        )
    (root / "_annotations.coco.json").write_text(json.dumps(coco))
    return root


def _write_preds(path: Path, rows: list[tuple]) -> Path:
    lines = ["file,class_name,confidence,x,y,w,h"]
    lines += [",".join(str(v) for v in r) for r in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


PLATE = (40.0, 40.0, 16.0, 60.0)
NEXT_TO = (56.0, 30.0, 20.0, 80.0)


def test_measure_split_reads_outcomes_and_backdrops(tmp_path: Path) -> None:
    split = _write_split(
        tmp_path / "test",
        [
            # misread plate on yellow; a blue 20 kg plate beside it must not count as backdrop
            {"name": "a.jpg", "bg": YELLOW, "boxes": [(2, PLATE, GREY), (3, NEXT_TO, BLUE)]},
            {"name": "b.jpg", "bg": BLUE, "boxes": [(2, PLATE, GREY)]},  # read correctly
            {"name": "c.jpg", "bg": GREY, "boxes": [(2, PLATE, GREY)]},  # missed
            {"name": "d.jpg", "bg": GREY, "boxes": [(3, PLATE, GREY)]},  # no 1.5 kg: skipped
        ],
    )
    preds = _write_preds(
        tmp_path / "p.csv",
        [
            ("a.jpg", "0.5kg", 0.95, *PLATE),
            ("b.jpg", "1.5kg", 0.95, *PLATE),
            ("c.jpg", "1.5kg", 0.50, *PLATE),  # below 0.80 → dropped
        ],
    )
    got = {m.image: m for m in measure_split(split, preds)}
    assert set(got) == {"a.jpg", "b.jpg", "c.jpg"}
    assert (got["a.jpg"].outcome, got["b.jpg"].outcome, got["c.jpg"].outcome) == (
        "0.5kg",
        "1.5kg",
        MISSED,
    )
    yellow_b = srgb_to_lab(np.array(YELLOW, dtype=np.uint8))[2]
    blue_b = srgb_to_lab(np.array(BLUE, dtype=np.uint8))[2]
    assert got["a.jpg"].ring_b == pytest.approx(yellow_b)  # neighbour masked out
    assert got["b.jpg"].ring_b == pytest.approx(blue_b)
    assert got["c.jpg"].plate_chroma == pytest.approx(0.0, abs=0.01)


def test_black_padding_is_not_backdrop_and_small_rings_are_unmeasurable(tmp_path: Path) -> None:
    split = _write_split(
        tmp_path / "test",
        [
            {"name": "pad.jpg", "bg": (0, 0, 0), "boxes": [(2, PLATE, GREY)]},
            {"name": "full.jpg", "bg": YELLOW, "boxes": [(2, (0.0, 0.0, 120.0, 160.0), GREY)]},
        ],
    )
    preds = _write_preds(tmp_path / "p.csv", [])
    for m in measure_split(split, preds):
        assert m.ring_pixels < MIN_RING_PIXELS
        assert m.ring_b is None


def test_pixel_size_mismatch_stops_the_run(tmp_path: Path) -> None:
    split = _write_split(
        tmp_path / "test",
        [{"name": "a.jpg", "bg": GREY, "boxes": [(2, PLATE, GREY)], "coco_w": 99}],
    )
    with pytest.raises(ValueError, match="would not line up"):
        measure_split(split, _write_preds(tmp_path / "p.csv", []))


def test_auc_counts_ties_as_half() -> None:
    assert auc([3.0, 2.0], [1.0, 2.0]) == pytest.approx(0.875)
    assert np.isnan(auc([], [1.0]))


def test_cluster_bootstrap_is_seeded_and_resamples_whole_images() -> None:
    by_image = {f"m{i}": ([10.0 + i], []) for i in range(8)} | {
        f"c{i}": ([], [float(i)]) for i in range(8)
    }
    ci, skipped = cluster_bootstrap_auc(by_image, reps=200)
    assert ci == (1.0, 1.0)  # perfectly separated in every resample
    assert cluster_bootstrap_auc(by_image, reps=200) == (ci, skipped)


N_OK = {M: 20, C: 20}
IMG_OK = {M: 8, C: 8}


@pytest.mark.parametrize(
    ("a", "ci", "expected"),
    [
        (0.80, (0.62, 0.93), "for"),
        (0.72, (0.52, 0.88), "inconclusive"),  # lower bound not above 0.55
        (0.20, (0.08, 0.40), "reversed"),
        (0.52, (0.35, 0.68), "against"),
        (0.55, (0.28, 0.80), "inconclusive"),  # too wide to rule anything out
    ],
)
def test_decision_rule(a, ci, expected) -> None:
    assert decide(N_OK, IMG_OK, a, ci) == expected


def test_too_few_boxes_or_images_is_inconclusive_whatever_the_auc() -> None:
    assert decide({M: 14, C: 20}, IMG_OK, 0.95, (0.9, 1.0)) == "inconclusive"
    assert decide(N_OK, {M: 8, C: 5}, 0.95, (0.9, 1.0)) == "inconclusive"


def test_summarise_counts_groups_and_within_image_pairs() -> None:
    ms = [
        BoxMeasure("x", "0.5kg", 500, 30.0, 5.0),
        BoxMeasure("x", "1.5kg", 500, 10.0, 40.0),
        BoxMeasure("y", "1.5kg", 500, 20.0, 40.0),
        BoxMeasure("y", MISSED, 500, 0.0, 10.0),
        BoxMeasure("z", "0.5kg", 20, None, 5.0),
    ]
    s = summarise(ms, reps=50)
    assert s["measurable_boxes"][M] == 1 and s["measurable_boxes"][C] == 2
    assert s["unmeasurable_boxes"][M] == 1
    assert s["auc_misread_vs_correct"] == 1.0
    assert s["within_image"] == {"images_with_both": 1, "pairs": 1, "auc": 1.0}
    assert s["verdict"] == "inconclusive"  # far below the minimum n
