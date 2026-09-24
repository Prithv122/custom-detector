from datetime import datetime, timedelta

from detector.sort_photos import group_by_gap


def _times(*offsets_seconds: float) -> list[datetime]:
    base = datetime(2026, 9, 24, 9, 0, 0)
    return [base + timedelta(seconds=s) for s in offsets_seconds]


def test_single_burst_stays_together():
    timestamps = _times(0, 5, 10, 15)
    assert group_by_gap(timestamps, gap_seconds=120) == [[0, 1, 2, 3]]


def test_gap_over_threshold_starts_new_group():
    timestamps = _times(0, 5, 10, 200, 205, 210)
    assert group_by_gap(timestamps, gap_seconds=120) == [[0, 1, 2], [3, 4, 5]]


def test_out_of_order_input_is_sorted_first():
    timestamps = _times(10, 0, 5)
    assert group_by_gap(timestamps, gap_seconds=120) == [[1, 2, 0]]


def test_empty_input():
    assert group_by_gap([], gap_seconds=120) == []


def test_sort_by_slate_copies_into_per_load_folders(tmp_path):
    from detector.sort_photos import sort_by_slate

    source = tmp_path / "phone_dump"
    source.mkdir()
    (source / "IMG_20260924_090000.jpg").write_bytes(b"slate")
    (source / "IMG_20260924_090010.jpg").write_bytes(b"photo1")
    (source / "IMG_20260924_090500.jpg").write_bytes(b"slate2")

    dest = tmp_path / "gym_a"
    answers = iter(["L01", "L02"])
    moved = sort_by_slate(source, dest, gap_seconds=60, ask_load_id=lambda *_: next(answers))

    assert len(moved) == 3
    assert sorted(p.name for p in (dest / "L01").iterdir()) == ["L01_01.jpg", "L01_02.jpg"]
    assert sorted(p.name for p in (dest / "L02").iterdir()) == ["L02_01.jpg"]
    assert (dest / "manifest.csv").exists()


def test_sort_by_slate_skips_blank_answer(tmp_path):
    from detector.sort_photos import sort_by_slate

    source = tmp_path / "phone_dump"
    source.mkdir()
    (source / "IMG_20260924_090000.jpg").write_bytes(b"slate")

    dest = tmp_path / "gym_a"
    moved = sort_by_slate(source, dest, ask_load_id=lambda *_: "")

    assert moved == []
