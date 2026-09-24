from pathlib import Path

from detector.protocol import LOOSE_LOAD_ID, VALID_PLATE_CLASSES_KG, parse_load_schedule

PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "data" / "CAPTURE_PROTOCOL.md"


def test_parses_all_24_loads_plus_loose():
    schedule = parse_load_schedule(PROTOCOL_PATH)
    assert {f"L{n:02d}" for n in range(1, 25)} | {LOOSE_LOAD_ID} == set(schedule)


def test_split_counts_match_18_3_3():
    schedule = parse_load_schedule(PROTOCOL_PATH)
    bar_loads = {k: v for k, v in schedule.items() if k != LOOSE_LOAD_ID}
    counts = {"train": 0, "val": 0, "test": 0}
    for spec in bar_loads.values():
        counts[spec.split] += 1
    assert counts == {"train": 18, "val": 3, "test": 3}


def test_val_and_test_cover_every_plate_class():
    schedule = parse_load_schedule(PROTOCOL_PATH)
    for split in ("val", "test"):
        classes = {p for spec in schedule.values() if spec.split == split for p in spec.per_side_kg}
        assert classes == VALID_PLATE_CLASSES_KG


def test_l07_parses_stacked_plates():
    schedule = parse_load_schedule(PROTOCOL_PATH)
    assert schedule["L07"].per_side_kg == (25.0, 2.5)
    assert schedule["L07"].total_kg == 75.0


def test_loose_is_train_with_no_ground_truth():
    schedule = parse_load_schedule(PROTOCOL_PATH)
    assert schedule[LOOSE_LOAD_ID].split == "train"
    assert schedule[LOOSE_LOAD_ID].per_side_kg == ()
