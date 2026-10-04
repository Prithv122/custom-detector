"""Local demo: upload a photo of a loaded sleeve and see each plate's weight.

Runs the published 2A checkpoint (Hugging Face Hub, pinned to one commit, SHA-256 checked before
the pickled file is loaded) on CPU. Fetching, thresholding and drawing have no Gradio import, so
they are testable without it; ``build_app`` imports Gradio when it is called.

The model is run once per photo at the evaluation floor (``recolour_run.PRED_THRESHOLD``) and the
confidence slider only filters those raw detections, the same way the evaluation applies its cut.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from detector.dataset import CLASSES
from detector.evaluate import Detection
from detector.recolour_run import load_model, predict, sha256_file

REPO_ID = "Prithv122/custom-detector"
FILENAME = "checkpoint_best_total.pth"
REVISION = "be16623ab1de46cc2e2b9e4df477af5d71d04c58"
EXPECTED_SHA256 = "f802956130702712f05bfb82d8b437ea1648efe642ff90bbba2fb38a1d8bce1f"
DEFAULT_THRESHOLD = 0.53  # 2A's val-chosen threshold (results/run2_evaluation.json)
MIN_THRESHOLD, MAX_THRESHOLD = 0.01, 0.99

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
TABLE_HEADERS = ["class", "confidence", "x0", "y0", "x1", "y1"]

# One colour per class, fixed so a weight keeps its colour across photos.
PALETTE = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 160, 0),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (0, 160, 160),
    (200, 50, 200),
    (128, 96, 0),
    (0, 0, 128),
    (90, 90, 90),
]
CLASS_COLOURS = dict(zip(CLASSES, PALETTE, strict=True))

NOTES = f"""\
## Weightlifting plate detector

Names the weight of each plate (0.5 to 25 kg) and the collar (`zacisk`) in a **side-on photo of
one loaded barbell sleeve**. RF-DETR-S fine-tuned on a public dataset
([ProjektCiezary, CC BY 4.0](https://universe.roboflow.com/projektciezary/weightlifting-plates),
cleaned and re-split). The photo is processed on this machine and not uploaded anywhere.

**Confidence threshold.** {DEFAULT_THRESHOLD} is the value chosen on the validation split for this
checkpoint and the setting its test numbers were measured at. Other values are for exploring.

**Known limits**
- On the held-out test day, 25 of 100 plates labelled 1.5 kg were still read as 0.5 kg
  (72 read correctly, 3 as another class, 0 missed) at {DEFAULT_THRESHOLD}. The error followed
  scene colour, and a colour-cast augmentation reduced it from 51 to 25 of 100 in a controlled
  run. It is reduced, not removed.
- At {DEFAULT_THRESHOLD} the model also makes 12 test-day predictions that match no labelled box.
- Trained on side-on close-ups of one sleeve. Wide shots, other angles or several bars in one
  frame are outside what it was trained and tested on.
- The weights of the detected plates are **not** summed: no independent check of a loaded total
  was ever made.

[Model card](https://huggingface.co/Prithv122/custom-detector/tree/{REVISION}) ·
[Code and results](https://github.com/Prithv122/custom-detector)
"""


# --- checkpoint ----------------------------------------------------------------------------


def verify_checkpoint(path: Path) -> Path:
    """Return ``path`` if its SHA-256 is the published 2A checkpoint's, else raise."""
    got = sha256_file(path)
    if got != EXPECTED_SHA256:
        raise RuntimeError(
            f"{path} is not the published 2A checkpoint (SHA-256 {got}, expected {EXPECTED_SHA256})"
        )
    return path


def fetch_checkpoint(cache_dir: str | Path | None = None) -> Path:
    """Download the 2A checkpoint at the pinned Hub revision and check its hash."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO_ID, FILENAME, revision=REVISION, cache_dir=cache_dir)
    return verify_checkpoint(Path(path))


def load(checkpoint: Path | None = None) -> Any:
    """Load the model on CPU from ``checkpoint`` or, by default, from the pinned Hub revision."""
    path = verify_checkpoint(checkpoint) if checkpoint else fetch_checkpoint()
    return load_model(path, device="cpu")


# --- inference and drawing -----------------------------------------------------------------


def prepare_image(image: Image.Image) -> Image.Image:
    """RGB with the EXIF rotation applied (phone photos are often stored sideways)."""
    return ImageOps.exif_transpose(image).convert("RGB")


def detect(model: Any, image: Image.Image) -> list[Detection]:
    """Every raw detection above the evaluation floor; filter with ``visible``."""
    import numpy as np

    return predict(model, np.asarray(image))


def visible(dets: list[Detection], threshold: float) -> list[Detection]:
    """Detections at or above ``threshold`` (as the evaluation applies it), best first."""
    return sorted((d for d in dets if d.confidence >= threshold), key=lambda d: -d.confidence)


def draw(image: Image.Image, dets: list[Detection]) -> Image.Image:
    """A copy of ``image`` with each detection's box and ``class confidence`` label."""
    out = image.copy()
    canvas = ImageDraw.Draw(out)
    scale = max(out.size) / 800
    width = max(2, round(2 * scale))
    font = ImageFont.load_default(size=max(12, round(14 * scale)))
    for d in dets:
        x, y, w, h = d.box
        colour = CLASS_COLOURS.get(d.cls, (0, 0, 0))
        canvas.rectangle((x, y, x + w, y + h), outline=colour, width=width)
        label = f"{d.cls} {d.confidence:.2f}"
        left, top, right, bottom = canvas.textbbox((x, y), label, font=font)
        top_edge = max(0, top - 2 * width)
        canvas.rectangle((left, top_edge, right + width, bottom + width), fill=colour)
        canvas.text((left + width // 2, top_edge + width // 2), label, fill="white", font=font)
    return out


def table_rows(dets: list[Detection]) -> list[list]:
    rows = []
    for d in dets:
        x, y, w, h = d.box
        rows.append([d.cls, round(d.confidence, 3), round(x), round(y), round(x + w), round(y + h)])
    return rows


def summarise(dets: list[Detection], threshold: float) -> str:
    if not dets:
        return f"No detections at confidence ≥ {threshold:.2f}."
    counts: dict[str, int] = {}
    for d in dets:
        counts[d.cls] = counts.get(d.cls, 0) + 1
    ordered = [c for c in CLASSES if c in counts]
    parts = ", ".join(f"{counts[c]} x {c}" for c in ordered)
    return f"{len(dets)} detections at confidence ≥ {threshold:.2f}: {parts}."


def render(
    image: Image.Image, dets: list[Detection], threshold: float
) -> tuple[Image.Image, str, list[list]]:
    shown = visible(dets, threshold)
    return draw(image, shown), summarise(shown, threshold), table_rows(shown)


# --- app -----------------------------------------------------------------------------------


def example_images(directory: Path, limit: int = 6) -> list[str]:
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return [str(p) for p in files[:limit]]


def build_app(model: Any, threshold: float = DEFAULT_THRESHOLD, examples: list[str] | None = None):
    """The Gradio app. ``model`` needs only ``.predict(PIL image, threshold=...)``."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError("the demo needs Gradio: run `uv sync --extra demo`") from exc

    def on_image(image: Image.Image | None, thr: float):
        if image is None:
            return None, "", [], None
        prepared = prepare_image(image)
        dets = detect(model, prepared)
        return (*render(prepared, dets, thr), (prepared, dets))

    def on_threshold(state: tuple[Image.Image, list[Detection]] | None, thr: float):
        if state is None:
            return None, "", []
        return render(state[0], state[1], thr)

    with gr.Blocks(title="Weightlifting plate detector", analytics_enabled=False) as app:
        gr.Markdown(NOTES)
        with gr.Row():
            with gr.Column():
                photo = gr.Image(type="pil", label="Photo of a loaded sleeve", sources=["upload"])
                slider = gr.Slider(
                    MIN_THRESHOLD,
                    MAX_THRESHOLD,
                    value=threshold,
                    step=0.01,
                    label="Confidence threshold",
                )
                if examples:
                    gr.Examples(examples=examples, inputs=photo, label="Example photos")
            with gr.Column():
                shown = gr.Image(label="Detections", interactive=False)
                summary = gr.Markdown()
                table = gr.Dataframe(headers=TABLE_HEADERS, interactive=False, label="Detections")
        state = gr.State()
        # Re-apply the slider once inference is done, so a threshold moved while the model was
        # running still decides what is shown.
        photo.change(
            on_image, [photo, slider], [shown, summary, table, state], api_name="detect"
        ).then(on_threshold, [state, slider], [shown, summary, table], api_name=False)
        slider.change(on_threshold, [state, slider], [shown, summary, table], api_name=False)
    return app
