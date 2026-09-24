from pathlib import Path

from detector.protocol import parse_load_schedule
from detector.roboflow_upload import resolve_split, upload_gym_a

PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "data" / "CAPTURE_PROTOCOL.md"


class FakeProject:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, list[str]]] = []

    def upload(self, image_path: str, split: str, tag_names: list[str]) -> None:
        self.calls.append((image_path, split, tag_names))


def test_resolve_split_matches_the_pre_fixed_schedule():
    schedule = parse_load_schedule(PROTOCOL_PATH)
    assert resolve_split("L08", schedule) == "val"
    assert resolve_split("L09", schedule) == "test"
    assert resolve_split("L01", schedule) == "train"
    assert resolve_split("LOOSE", schedule) == "train"


def test_upload_gym_a_routes_each_load_folder_by_split(tmp_path):
    (tmp_path / "L08").mkdir()
    (tmp_path / "L08" / "L08_01.jpg").write_bytes(b"fake")
    (tmp_path / "L09").mkdir()
    (tmp_path / "L09" / "L09_01.jpg").write_bytes(b"fake")
    (tmp_path / "LOOSE").mkdir()
    (tmp_path / "LOOSE" / "loose_01.jpg").write_bytes(b"fake")

    project = FakeProject()
    result = upload_gym_a(tmp_path, PROTOCOL_PATH, project)

    assert result.uploaded == {"val": 1, "test": 1, "train": 1}
    assert not result.failures
    assert ("L08" in call[2] for call in project.calls)


def test_upload_gym_a_records_unknown_load_id_as_a_failure(tmp_path):
    (tmp_path / "L99").mkdir()
    (tmp_path / "L99" / "x.jpg").write_bytes(b"fake")

    project = FakeProject()
    result = upload_gym_a(tmp_path, PROTOCOL_PATH, project)

    assert result.uploaded == {}
    assert len(result.failures) == 1
    assert "not in the capture protocol schedule" in result.failures[0][1]


def test_upload_gym_a_continues_after_a_single_upload_failure(tmp_path):
    (tmp_path / "L01").mkdir()
    (tmp_path / "L01" / "a.jpg").write_bytes(b"fake")
    (tmp_path / "L01" / "b.jpg").write_bytes(b"fake")

    class FlakyProject:
        def __init__(self) -> None:
            self.calls = 0

        def upload(self, image_path: str, split: str, tag_names: list[str]) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("network error")

    result = upload_gym_a(tmp_path, PROTOCOL_PATH, FlakyProject())

    assert result.uploaded == {"train": 1}
    assert len(result.failures) == 1
