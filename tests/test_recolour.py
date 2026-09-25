"""Recolouring transform on synthetic images whose answer is known before measuring.

No inference here — these tests only check the transform itself, per the order in NOTES.md
("code + synthetic-image tests" comes before the baseline gate and the one run).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from detector.backdrop import MIN_RING_PIXELS, srgb_to_lab
from detector.recolour import (
    ARMS,
    KEEP,
    REVERSE,
    RING,
    SHAM,
    WHOLE,
    RingColour,
    apply_arm,
    apply_recolour,
    lab_to_srgb,
    measure_ring_ab,
    target_colours,
    whole_mask,
)

YELLOW, BLUE, GREY, WHITE_BG = (230, 220, 40), (20, 40, 150), (128, 128, 128), (240, 240, 240)
CATS = [{"id": 1, "name": "0.5kg"}, {"id": 2, "name": "1.5kg"}]
PLATE = (40.0, 40.0, 16.0, 60.0)


@pytest.mark.parametrize(
    "rgb",
    [(255, 255, 255), (0, 0, 0), (255, 255, 0), (0, 0, 255), (230, 220, 40), (128, 64, 200)],
)
def test_lab_to_srgb_inverts_srgb_to_lab(rgb) -> None:
    lab = srgb_to_lab(np.array(rgb, dtype=np.uint8))
    back = lab_to_srgb(lab)
    assert back == pytest.approx(rgb, abs=0.05)


def test_mask_is_the_only_place_pixels_change() -> None:
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    rgb[:] = YELLOW
    rgb[5:10, 5:10] = GREY  # a "plate" the mask will exclude
    mask = np.zeros((20, 20), dtype=bool)
    mask[0:12, 0:12] = True
    mask[5:10, 5:10] = False  # simulate a plate carved out of the ring

    out, _ = apply_recolour(rgb, mask, (0.0, -40.0))  # push toward blue

    assert np.array_equal(out[~mask], rgb[~mask])  # untouched, bit-identical
    assert not np.array_equal(out[mask], rgb[mask])  # every masked pixel moved


def test_l_star_is_kept_within_rounding() -> None:
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb[:] = YELLOW
    mask = np.ones((10, 10), dtype=bool)
    before = srgb_to_lab(rgb)[..., 0]

    out, clipped = apply_recolour(rgb, mask, (5.0, -20.0))  # modest shift, stays in gamut

    after = srgb_to_lab(out)[..., 0]
    assert clipped == 0.0
    assert np.abs(after - before).max() < 1.0  # only 8-bit rounding, no drift from the recolour


@pytest.mark.parametrize("rgb", [YELLOW, BLUE, GREY, WHITE_BG, (10, 200, 90)])
def test_sham_round_trip_moves_no_channel_by_more_than_one(rgb) -> None:
    arr = np.zeros((6, 6, 3), dtype=np.uint8)
    arr[:] = rgb
    mask = np.ones((6, 6), dtype=bool)

    out, clipped = apply_recolour(arr, mask, None)

    assert np.abs(out.astype(int) - arr.astype(int)).max() <= 1
    assert clipped == 0.0


def test_whole_mask_excludes_every_box_and_black_padding() -> None:
    rgb = np.full((20, 30, 3), 100, dtype=np.uint8)
    rgb[0:5, 0:30] = 0  # letterbox padding
    boxes = [(2.0, 2.0, 4.0, 4.0), (10.0, 10.0, 5.0, 5.0)]

    mask = whole_mask(boxes, (20, 30), rgb)

    assert not mask[0:5, :].any()  # black padding excluded
    assert not mask[2:6, 2:6].any()  # first box excluded
    assert not mask[10:15, 10:15].any()  # second box excluded
    assert mask[15, 20]  # plain background pixel is in


def test_arms_table_matches_the_protocol() -> None:
    assert ARMS[RING] == ("ring", "blue")
    assert ARMS[SHAM] == ("ring", None)
    assert ARMS[WHOLE] == ("whole", "blue")
    assert ARMS[REVERSE] == ("ring", "yellow")
    assert ARMS[KEEP] == ("ring", "blue")


def test_apply_arm_uses_the_right_mask_and_target() -> None:
    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    rgb[:] = YELLOW
    box = (10.0, 10.0, 10.0, 10.0)
    all_boxes = [box]
    targets = {"blue": (0.0, -60.0), "yellow": (0.0, 60.0)}

    ring_out, _ = apply_arm(rgb, box, all_boxes, RING, targets)
    whole_out, _ = apply_arm(rgb, box, all_boxes, WHOLE, targets)
    sham_out, _ = apply_arm(rgb, box, all_boxes, SHAM, targets)
    reverse_out, _ = apply_arm(rgb, box, all_boxes, REVERSE, targets)

    # Ring: touches the area right around the box but never the box itself.
    assert np.array_equal(ring_out[10:20, 10:20], rgb[10:20, 10:20])
    assert not np.array_equal(ring_out[0:10, 0:10], rgb[0:10, 0:10])
    # Whole: touches everywhere except the box.
    assert np.array_equal(whole_out[10:20, 10:20], rgb[10:20, 10:20])
    assert not np.array_equal(whole_out[0:5, 0:5], rgb[0:5, 0:5])
    # Sham: no colour change anywhere.
    assert np.abs(sham_out.astype(int) - rgb.astype(int)).max() <= 1
    # Ring (blue target) and Reverse (yellow target) push opposite ways, measured inside the ring.
    ring_b = srgb_to_lab(ring_out[5, 5])[2]
    reverse_b = srgb_to_lab(reverse_out[5, 5])[2]
    assert ring_b < 0 < reverse_b


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
                "width": w,
                "height": h,
                "extra": {"name": spec["name"]},
            }
        )
    (root / "_annotations.coco.json").write_text(json.dumps(coco))
    return root


def test_measure_ring_ab_is_outcome_blind_and_skips_small_rings(tmp_path: Path) -> None:
    split = _write_split(
        tmp_path / "test",
        [
            {"name": "yellow.jpg", "bg": YELLOW, "boxes": [(2, PLATE, GREY)]},
            {"name": "blue.jpg", "bg": BLUE, "boxes": [(2, PLATE, GREY)]},
            {"name": "no_target.jpg", "bg": GREY, "boxes": [(1, PLATE, GREY)]},  # skipped
            {
                "name": "full.jpg",  # box fills the frame -> unmeasurable ring
                "bg": YELLOW,
                "boxes": [(2, (0.0, 0.0, 120.0, 160.0), GREY)],
            },
        ],
    )

    rows = measure_ring_ab(split)

    assert {r.image for r in rows} == {"yellow.jpg", "blue.jpg", "full.jpg"}
    by_name = {r.image: r for r in rows}
    yellow_b = srgb_to_lab(np.array(YELLOW, dtype=np.uint8))[2]
    blue_b = srgb_to_lab(np.array(BLUE, dtype=np.uint8))[2]
    assert by_name["yellow.jpg"].ring_b == pytest.approx(yellow_b)
    assert by_name["blue.jpg"].ring_b == pytest.approx(blue_b)
    assert by_name["full.jpg"].ring_pixels < MIN_RING_PIXELS
    assert by_name["full.jpg"].ring_a is None and by_name["full.jpg"].ring_b is None


def test_pixel_size_mismatch_stops_the_run(tmp_path: Path) -> None:
    root = tmp_path / "test"
    root.mkdir()
    coco = {
        "categories": CATS,
        "images": [{"id": 0, "file_name": "a.png", "width": 99, "height": 160, "extra": {}}],
        "annotations": [{"id": 0, "image_id": 0, "category_id": 2, "bbox": list(PLATE)}],
    }
    arr = np.full((160, 120, 3), GREY, dtype=np.uint8)
    Image.fromarray(arr).save(root / "a.png")
    (root / "_annotations.coco.json").write_text(json.dumps(coco))

    with pytest.raises(ValueError, match="would not line up"):
        measure_ring_ab(root)


def test_targets_come_out_of_the_rule() -> None:
    rows = [
        RingColour("a", 500, 10.0, -20.0),  # blue side
        RingColour("b", 500, 20.0, -10.0),  # blue side
        RingColour("c", 500, 0.0, 50.0),  # yellow side
        RingColour("d", 500, 4.0, 40.0),  # yellow side
        RingColour("e", 500, 999.0, 0.0),  # b* == 0 counts as blue side (rule's neutral point)
        RingColour("f", 20, None, None),  # unmeasurable, excluded
    ]

    t = target_colours(rows)

    blue_a, blue_b = np.median([10.0, 20.0, 999.0]), np.median([-20.0, -10.0, 0.0])
    assert t["blue"] == pytest.approx((blue_a, blue_b))
    assert t["yellow"] == pytest.approx((2.0, 45.0))
    assert t["n_blue"] == 3
    assert t["n_yellow"] == 2
    assert t["n_unmeasurable"] == 1


def test_target_colours_with_no_boxes_on_a_side_is_nan() -> None:
    t = target_colours([RingColour("a", 500, 1.0, 1.0)])
    assert t["n_blue"] == 0
    assert np.isnan(t["blue"][0]) and np.isnan(t["blue"][1])
