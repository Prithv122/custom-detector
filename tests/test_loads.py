import csv
from pathlib import Path

from detector.loads import REQUIRED_COLUMNS, validate_loads_csv

PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "data" / "CAPTURE_PROTOCOL.md"


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _row(**overrides: str) -> dict[str, str]:
    base = {
        "load_id": "L01",
        "gym": "gym_a",
        "date": "2026-09-24",
        "bar_kg": "20",
        "collar_kg_each": "0",
        "side_plates_kg": "25",
        "total_kg": "70",
        "photos": "6",
        "notes": "",
    }
    base.update(overrides)
    return base


def test_empty_csv_has_no_errors_but_warns_everything_missing(tmp_path):
    csv_path = tmp_path / "loads.csv"
    _write_csv(csv_path, [])

    report = validate_loads_csv(csv_path, PROTOCOL_PATH)

    assert report.ok
    assert any("not yet recorded" in w.message for w in report.warnings)


def test_correct_row_passes(tmp_path):
    csv_path = tmp_path / "loads.csv"
    _write_csv(csv_path, [_row(load_id="L07", side_plates_kg="25+2.5", total_kg="75")])

    report = validate_loads_csv(csv_path, PROTOCOL_PATH)

    assert report.ok
    assert all("not yet recorded" in w.message for w in report.warnings)


def test_wrong_total_is_an_error(tmp_path):
    csv_path = tmp_path / "loads.csv"
    _write_csv(csv_path, [_row(total_kg="999")])

    report = validate_loads_csv(csv_path, PROTOCOL_PATH)

    assert not report.ok
    assert any("doesn't match" in e.message for e in report.errors)


def test_invalid_plate_class_is_an_error(tmp_path):
    csv_path = tmp_path / "loads.csv"
    _write_csv(csv_path, [_row(side_plates_kg="1.25", total_kg="22.5")])

    report = validate_loads_csv(csv_path, PROTOCOL_PATH)

    assert not report.ok
    assert any("outside the six allowed classes" in e.message for e in report.errors)


def test_unknown_load_id_is_an_error(tmp_path):
    csv_path = tmp_path / "loads.csv"
    _write_csv(csv_path, [_row(load_id="L99")])

    report = validate_loads_csv(csv_path, PROTOCOL_PATH)

    assert not report.ok
    assert any("not in the capture protocol schedule" in e.message for e in report.errors)


def test_substituted_plates_is_a_warning_not_an_error(tmp_path):
    csv_path = tmp_path / "loads.csv"
    _write_csv(
        csv_path,
        [_row(side_plates_kg="20+5", total_kg="70", notes="substituted, no spare 25kg plate")],
    )

    report = validate_loads_csv(csv_path, PROTOCOL_PATH)

    assert report.ok
    assert any("differ from the schedule" in w.message for w in report.warnings)


def test_wrong_photo_count_is_a_warning(tmp_path):
    csv_path = tmp_path / "loads.csv"
    _write_csv(csv_path, [_row(photos="4")])

    report = validate_loads_csv(csv_path, PROTOCOL_PATH)

    assert report.ok
    assert any("protocol calls for 6" in w.message for w in report.warnings)


def test_header_mismatch_is_a_file_level_error(tmp_path):
    csv_path = tmp_path / "loads.csv"
    csv_path.write_text("wrong,header\n", encoding="utf-8")

    report = validate_loads_csv(csv_path, PROTOCOL_PATH)

    assert not report.ok
    assert report.errors[0].row is None
