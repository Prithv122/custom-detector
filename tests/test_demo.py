"""The demo's logic: checkpoint pinning, thresholding, drawing and the app wiring."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from detector import demo
from detector.cli import build_parser
from detector.dataset import CLASSES
from detector.evaluate import Detection
from detector.recolour_run import PRED_THRESHOLD

ROOT = Path(__file__).resolve().parents[1]


def det(cls: str, conf: float, box=(10.0, 10.0, 20.0, 20.0)) -> Detection:
    return Detection(cls, conf, box)


# --- pinned constants agree with the committed evidence -------------------------------------


def test_default_threshold_is_the_val_chosen_one() -> None:
    run2 = json.loads((ROOT / "results" / "run2_evaluation.json").read_text())
    assert run2["runs"]["2A"]["threshold"] == demo.DEFAULT_THRESHOLD


def test_pinned_revision_and_hash_match_the_publication_record() -> None:
    record = json.loads((ROOT / "results" / "hf_publication.json").read_text())
    assert record["revision"] == demo.REVISION
    assert record["sha256_expected"] == demo.EXPECTED_SHA256
    assert record["file"] == demo.FILENAME
    assert re.fullmatch(r"[0-9a-f]{40}", demo.REVISION)


def test_every_class_has_a_colour() -> None:
    assert set(demo.CLASS_COLOURS) == set(CLASSES)


# --- checkpoint ------------------------------------------------------------------------------


def test_verify_checkpoint_rejects_a_different_file(tmp_path: Path) -> None:
    f = tmp_path / "x.pth"
    f.write_bytes(b"not the checkpoint")
    with pytest.raises(RuntimeError, match="not the published 2A checkpoint"):
        demo.verify_checkpoint(f)


def test_verify_checkpoint_accepts_a_matching_file(tmp_path: Path, monkeypatch) -> None:
    f = tmp_path / "x.pth"
    f.write_bytes(b"stand-in")
    monkeypatch.setattr(demo, "sha256_file", lambda p: demo.EXPECTED_SHA256)
    assert demo.verify_checkpoint(f) == f


def test_fetch_checkpoint_pins_repo_file_and_revision(tmp_path: Path, monkeypatch) -> None:
    f = tmp_path / "x.pth"
    f.write_bytes(b"stand-in")
    seen = {}

    def fake_download(repo_id, filename, revision=None, cache_dir=None):
        seen.update(repo_id=repo_id, filename=filename, revision=revision)
        return str(f)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.setattr(demo, "sha256_file", lambda p: demo.EXPECTED_SHA256)
    assert demo.fetch_checkpoint() == f
    assert seen == {
        "repo_id": demo.REPO_ID,
        "filename": demo.FILENAME,
        "revision": demo.REVISION,
    }


# --- thresholding -----------------------------------------------------------------------------


def test_visible_keeps_the_threshold_itself_and_sorts_best_first() -> None:
    dets = [det("5kg", 0.60), det("25kg", 0.53), det("10kg", 0.52), det("zacisk", 0.99)]
    kept = demo.visible(dets, 0.53)
    assert [d.cls for d in kept] == ["zacisk", "5kg", "25kg"]  # >= as in the evaluation


def test_visible_is_empty_above_every_confidence() -> None:
    assert demo.visible([det("5kg", 0.6)], 0.9) == []


# --- inference wrapper ------------------------------------------------------------------------


class FakeModel:
    def __init__(self) -> None:
        self.thresholds: list[float] = []

    def predict(self, image, threshold):
        self.thresholds.append(threshold)
        return SimpleNamespace(
            xyxy=np.array([[5.0, 6.0, 25.0, 46.0]]),
            confidence=np.array([0.8]),
            data={"class_name": np.array(["25kg"])},
        )


def test_detect_runs_once_at_the_evaluation_floor_and_converts_boxes() -> None:
    model = FakeModel()
    dets = demo.detect(model, Image.new("RGB", (64, 48)))
    assert model.thresholds == [PRED_THRESHOLD]
    assert dets == [Detection("25kg", pytest.approx(0.8), (5.0, 6.0, 20.0, 40.0))]


def test_prepare_image_applies_exif_rotation_and_drops_alpha() -> None:
    im = Image.new("RGBA", (40, 20), (255, 0, 0, 128))
    exif = Image.Exif()
    exif[0x0112] = 6  # stored sideways; displayed rotated 90 degrees
    out = demo.prepare_image(_with_exif(im, exif))
    assert out.mode == "RGB"
    assert out.size == (20, 40)


def _with_exif(im: Image.Image, exif: Image.Exif) -> Image.Image:
    import io

    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", exif=exif)
    buf.seek(0)
    return Image.open(buf)


# --- drawing and text -------------------------------------------------------------------------


def test_draw_marks_the_box_and_leaves_the_input_alone() -> None:
    src = Image.new("RGB", (200, 150), (255, 255, 255))
    out = demo.draw(src, [det("25kg", 0.9, (40.0, 50.0, 100.0, 60.0))])
    assert out.size == src.size
    assert src.tobytes() == Image.new("RGB", (200, 150), (255, 255, 255)).tobytes()
    assert out.getpixel((40, 90)) == demo.CLASS_COLOURS["25kg"]  # left edge of the box
    assert out.getpixel((100, 80)) == (255, 255, 255)  # interior untouched


def test_draw_with_no_detections_is_a_plain_copy() -> None:
    src = Image.new("RGB", (30, 30), (10, 20, 30))
    out = demo.draw(src, [])
    assert out is not src
    assert out.tobytes() == src.tobytes()


def test_table_rows_use_corner_coordinates() -> None:
    rows = demo.table_rows([det("2kg", 0.91234, (10.4, 20.6, 30.0, 40.0))])
    assert rows == [["2kg", 0.912, 10, 21, 40, 61]]


def test_summary_counts_in_class_order_and_never_sums_weights() -> None:
    text = demo.summarise([det("zacisk", 0.9), det("25kg", 0.9), det("25kg", 0.8)], 0.53)
    assert text == "3 detections at confidence ≥ 0.53: 2 x 25kg, 1 x zacisk."
    assert "total" not in text.lower()
    assert "no detections" in demo.summarise([], 0.9).lower()


def test_render_filters_before_drawing_and_counting() -> None:
    src = Image.new("RGB", (100, 100), (255, 255, 255))
    dets = [det("25kg", 0.9), det("5kg", 0.2)]
    _, summary, rows = demo.render(src, dets, 0.53)
    assert summary.startswith("1 detections")
    assert [r[0] for r in rows] == ["25kg"]


def test_notes_quote_the_result_files() -> None:
    run2 = json.loads((ROOT / "results" / "run2_evaluation.json").read_text())
    a = run2["runs"]["2A"]
    counts = a["counts"]
    unmatched = sum(a["confusion"]["background"].values())
    text = " ".join(demo.NOTES.split())
    assert f"{demo.DEFAULT_THRESHOLD} is the value chosen" in text
    assert demo.REVISION in text
    assert f"{counts['0.5kg']} of 100 plates labelled 1.5 kg were still read as 0.5 kg" in text
    assert f"({counts['1.5kg']} read correctly, {counts['other']} as another class" in text
    assert f"{counts['missed']} missed)" in text
    assert f"from {run2['primary']['M_2C']} to {run2['primary']['M_2A']} of 100" in text
    assert f"also makes {unmatched} test-day predictions" in text


# --- examples and CLI -------------------------------------------------------------------------


def test_example_images_are_images_only_sorted_and_limited(tmp_path: Path) -> None:
    for name in ["b.jpg", "a.PNG", "c.webp", "notes.txt", "d.jpeg"]:
        (tmp_path / name).write_bytes(b"x")
    got = demo.example_images(tmp_path, limit=3)
    assert [Path(p).name for p in got] == ["a.PNG", "b.jpg", "c.webp"]


def test_cli_demo_defaults_to_the_documented_threshold() -> None:
    args = build_parser().parse_args(["demo"])
    assert args.threshold == demo.DEFAULT_THRESHOLD
    assert args.port == 7860


def test_cli_demo_rejects_an_out_of_range_threshold(capsys) -> None:
    args = build_parser().parse_args(["demo", "--threshold", "1.5"])
    assert args.func(args) == 1
    assert "--threshold must be between" in capsys.readouterr().err


# --- app wiring (needs the optional Gradio extra) -------------------------------------------


def test_app_builds_and_detect_endpoint_runs_the_model_once() -> None:
    pytest.importorskip("gradio")
    model = FakeModel()
    app = demo.build_app(model)
    assert app is not None
    fns = [fn for fn in app.fns.values() if getattr(fn, "api_name", None) == "detect"]
    assert len(fns) == 1
    out = fns[0].fn(Image.new("RGB", (64, 48), (255, 255, 255)), demo.DEFAULT_THRESHOLD)
    _, summary, rows, state = out
    assert len(model.thresholds) == 1
    assert summary.startswith("1 detections")
    assert rows[0][0] == "25kg"
    assert state[1][0].cls == "25kg"


def test_slider_and_photo_events_are_wired_to_the_right_triggers() -> None:
    pytest.importorskip("gradio")
    app = demo.build_app(FakeModel())
    by_name = {fn.fn.__name__: fn for fn in app.fns.values()}
    # `change`, not `release`: release fires only when the handle is dragged, not on typing.
    assert [t[1] for t in by_name["on_threshold"].targets].count("change") >= 1
    assert all(t[1] == "change" for t in by_name["on_threshold"].targets)
    assert all(t[1] == "change" for t in by_name["on_image"].targets)
    # on_threshold runs again after on_image, so a slider moved mid-inference still applies.
    assert len([fn for fn in app.fns.values() if fn.fn.__name__ == "on_threshold"]) == 2


def test_slider_callback_refilters_stored_detections_without_the_model() -> None:
    pytest.importorskip("gradio")
    model = FakeModel()
    app = demo.build_app(model)
    on_threshold = next(fn for fn in app.fns.values() if fn.fn.__name__ == "on_threshold").fn
    state = (Image.new("RGB", (64, 48), (255, 255, 255)), [det("25kg", 0.9), det("5kg", 0.6)])
    _, summary, rows = on_threshold(state, 0.7)
    assert summary == "1 detections at confidence ≥ 0.70: 1 x 25kg."
    assert [r[0] for r in rows] == ["25kg"]
    assert on_threshold(None, 0.7) == (None, "", [])
    assert model.thresholds == []
