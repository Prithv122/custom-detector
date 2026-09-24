from pathlib import Path

from detector.dataset.manifest import (
    DOUBLE_LABELLED,
    THIRD_PARTY_PRESS,
    THIRD_PARTY_VIDEO,
    UNLABELLED,
    build_manifest,
    exclusion_reason,
    parse_captured_at,
    read_manifest,
    write_manifest,
)

BOX = ("25kg", [0.0, 0.0, 10.0, 40.0])


def test_parse_captured_at_reads_phone_filenames() -> None:
    assert parse_captured_at("IMG_20251106_130044.jpg") == "2025-11-06T13:00:44"
    assert parse_captured_at("IMG_20251106_130044_jpg") == "2025-11-06T13:00:44"
    assert parse_captured_at("IMG_20250504_204553_BURST001_COVER.jpg") == "2025-05-04T20:45:53"
    assert parse_captured_at("ZE_210.png") == ""
    assert parse_captured_at("31.webp") == ""


def test_exclusion_rules() -> None:
    assert exclusion_reason("ZE_12.png", [BOX]) == THIRD_PARTY_VIDEO
    assert exclusion_reason("7.jpg", [BOX]) == THIRD_PARTY_PRESS
    assert exclusion_reason("IMG_20250501_184933.jpg", []) == UNLABELLED
    same_object_two_labels = [("0.5kg", [226, 236, 25, 169]), ("1.5kg", [227, 236, 25, 171])]
    assert exclusion_reason("IMG_20250908_113159_jpg", same_object_two_labels) == DOUBLE_LABELLED
    assert exclusion_reason("IMG_20250908_113159_jpg", [BOX]) == ""


def test_adjacent_plates_are_not_double_labelled() -> None:
    # Plates stacked on a sleeve overlap a little; that is not a labelling conflict.
    stacked = [("25kg", [100, 50, 30, 300]), ("20kg", [125, 60, 30, 280])]
    assert exclusion_reason("IMG_20250908_113159_jpg", stacked) == ""


def test_provenance_takes_precedence_over_label_problems() -> None:
    assert exclusion_reason("ZE_3.png", []) == THIRD_PARTY_VIDEO
    assert exclusion_reason("3.jpg", []) == THIRD_PARTY_PRESS


def test_build_manifest_reads_every_image(export: Path) -> None:
    records = build_manifest(export)
    by_file = {r.file: r for r in records}

    assert len(records) == 6
    assert by_file["IMG_20250501_120000.jpg"].boxes == "25kg:1;zacisk:1"
    assert by_file["IMG_20250501_120000.jpg"].published_split == "train"
    assert by_file["IMG_20250501_120000.jpg"].captured_at == "2025-05-01T12:00:00"
    assert by_file["ZE_7.png"].exclusion == THIRD_PARTY_VIDEO
    assert by_file["12.webp"].exclusion == THIRD_PARTY_PRESS
    assert by_file["IMG_20250501_120005.jpg"].exclusion == UNLABELLED
    assert by_file["IMG_20250502_090000.jpg"].exclusion == DOUBLE_LABELLED
    assert by_file["IMG_20250502_090500.jpg"].exclusion == ""
    assert all(len(r.phash256) == 64 and len(r.sha256) == 64 for r in records)


def test_manifest_round_trips_through_csv(export: Path, tmp_path: Path) -> None:
    records = build_manifest(export)
    records[0].split, records[0].session = "train", 3
    path = tmp_path / "manifest.csv"
    write_manifest(records, path)

    back = read_manifest(path)
    assert back == records
    assert back[0].box_counts == {"25kg": 1, "zacisk": 1}
    assert back[1].session is None
