"""Run 2 switch: both arms build through ``rfdetr``, the colour cast leaves boxes alone, and the
frozen-config check holds against Run 1's saved config.

No training and no GPU here. The behavioural claims come from running the real
``AlbumentationsWrapper`` on a synthetic neutral-grey image.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from rfdetr.config import AugmentationBackend, TrainConfig
from rfdetr.datasets.coco import make_coco_transforms_square_div_64
from rfdetr.datasets.transforms import AlbumentationsWrapper

from detector.backdrop import srgb_to_lab
from detector.run2 import (
    ALBUMENTATIONS_VERSION,
    ARMS,
    BACKEND,
    FLIP,
    PLANCKIAN,
    SEED,
    build_transforms,
    config_problems,
    get_arm,
)

ROOT = Path(__file__).resolve().parents[1]
RUN1 = json.loads((ROOT / "results" / "training_config.json").read_text())

# The notebook's TRAIN dict, as Run 1 used it.
TRAIN = dict(
    epochs=50,
    batch_size=8,
    grad_accum_steps=2,
    lr=1e-4,
    early_stopping=True,
    early_stopping_patience=10,
    run_test=True,
    log_per_class_metrics=True,
)


def test_arms_match_the_protocol() -> None:
    assert ARMS["2C"].aug_config == {"HorizontalFlip": {"p": 0.5}}
    assert ARMS["2A"].aug_config == {
        "HorizontalFlip": {"p": 0.5},
        "PlanckianJitter": {
            "mode": "cied",
            "temperature_limit": (4000, 15000),
            "sampling_method": "uniform",
            "p": 0.5,
        },
    }
    assert (ARMS["2C"].results_dir, ARMS["2A"].results_dir) == ("run2c", "run2a")
    assert (SEED, BACKEND, ALBUMENTATIONS_VERSION) == (20261001, "albumentations", "2.0.8")


def test_the_arms_differ_only_by_the_colour_transform() -> None:
    treatment = {k: v for k, v in ARMS["2A"].aug_config.items() if k != "PlanckianJitter"}
    assert treatment == ARMS["2C"].aug_config
    kw_c, kw_a = ARMS["2C"].train_kwargs(), ARMS["2A"].train_kwargs()
    assert {k: v for k, v in kw_c.items() if k != "aug_config"} == {
        k: v for k, v in kw_a.items() if k != "aug_config"
    }


def test_unknown_arm_is_refused() -> None:
    with pytest.raises(ValueError, match="RUN must be one of"):
        get_arm("1")


def test_train_kwargs_are_copies() -> None:
    kw = ARMS["2A"].train_kwargs()
    kw["aug_config"]["PlanckianJitter"]["p"] = 1.0
    kw["aug_config"].pop("HorizontalFlip")
    assert ARMS["2A"].aug_config["PlanckianJitter"]["p"] == 0.5
    assert "HorizontalFlip" in ARMS["2A"].aug_config


@pytest.mark.parametrize("key", ["2C", "2A"])
def test_arm_builds_through_rfdetr(key: str) -> None:
    arm = ARMS[key]
    wrappers = build_transforms(arm)
    inner = [w.transform.transforms[0] for w in wrappers]
    assert [type(t).__name__ for t in inner] == arm.transform_names
    assert inner[0].p == 0.5
    if key == "2A":
        jitter = inner[1]
        assert (jitter.mode, jitter.sampling_method, jitter.p) == ("cied", "uniform", 0.5)
        assert tuple(jitter.temperature_limit) == (4000, 15000)


@pytest.mark.parametrize("key", ["2C", "2A"])
def test_arm_takes_the_albumentations_training_path(key: str) -> None:
    """Both arms must leave rfdetr's torchvision pipeline, or the control is not a control."""
    arm = ARMS[key]
    cfg = TrainConfig(dataset_dir="ds", output_dir="out", **TRAIN, **arm.train_kwargs())
    assert (
        AugmentationBackend.from_str(cfg.model_dump()["augmentation_backend"])
        is AugmentationBackend.ALBU
    )
    pipeline = make_coco_transforms_square_div_64(
        "train", 512, multi_scale=True, expanded_scales=True, aug_config=arm.aug_config
    )
    names = [type(t).__name__ for t in pipeline.transforms]
    assert "RandomHorizontalFlip" not in names
    colour = [
        w
        for w in pipeline.transforms
        if isinstance(w, AlbumentationsWrapper)
        and type(w.transform.transforms[0]).__name__ == "PlanckianJitter"
    ]
    assert len(colour) == (1 if key == "2A" else 0)


def test_a_misspelt_transform_is_not_silently_skipped() -> None:
    """rfdetr's lenient build would drop it and train a second control; ours must raise."""
    typo = dataclasses.replace(
        ARMS["2A"], aug_config={**FLIP, "PlanckianJiter": PLANCKIAN["PlanckianJitter"]}
    )
    assert (
        AlbumentationsWrapper.from_config(copy.deepcopy(typo.aug_config)) != []
    )  # lenient: one left
    with pytest.raises(RuntimeError):
        build_transforms(typo)


# --- what the colour transform does ----------------------------------------------------------


def _always_on() -> AlbumentationsWrapper:
    params = {**ARMS["2A"].aug_config["PlanckianJitter"], "p": 1.0}
    return AlbumentationsWrapper.from_config({"PlanckianJitter": params}, strict=True)[0]


def test_colour_cast_keeps_boxes_and_labels() -> None:
    wrapper = _always_on()
    target = {
        "boxes": torch.tensor([[5.0, 6.0, 40.0, 50.0], [50.0, 10.0, 90.0, 60.0]]),
        "labels": torch.tensor([1, 2]),
    }
    image = Image.fromarray(np.full((64, 96, 3), 128, np.uint8))
    for _ in range(10):
        out, new = wrapper(image, {k: v.clone() for k, v in target.items()})
        assert out.size == image.size
        assert torch.equal(new["boxes"], target["boxes"])
        assert torch.equal(new["labels"], target["labels"])


def test_colour_cast_moves_chroma_far_more_than_lightness() -> None:
    wrapper = _always_on()
    grey = np.full((64, 96, 3), 128, np.uint8)
    base = srgb_to_lab(grey.astype(float))[0, 0]
    target = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.int64)}
    d_lightness, d_chroma = [], []
    for _ in range(60):
        out, _ = wrapper(Image.fromarray(grey), {k: v.clone() for k, v in target.items()})
        lab = srgb_to_lab(np.array(out).astype(float))[0, 0]
        d_lightness.append(abs(lab[0] - base[0]))
        d_chroma.append(float(np.hypot(*(lab[1:] - base[1:]))))
    # Protocol: L* moves at most 2.6; b* spans 74 across the range. Margins are loose on purpose.
    assert max(d_lightness) <= 3.0
    assert max(d_chroma) >= 25.0
    assert np.mean(d_chroma) >= 10 * np.mean(d_lightness)


# --- frozen config ---------------------------------------------------------------------------


def _preflight(key: str) -> dict:
    cfg = TrainConfig(
        dataset_dir="/tmp/plates_v10_clean",
        output_dir="/kaggle/working/run",
        **TRAIN,
        **ARMS[key].train_kwargs(),
    )
    return {"train_config": json.loads(cfg.model_dump_json())}


@pytest.mark.parametrize("key", ["2C", "2A"])
def test_only_the_allowed_fields_differ_from_run_1(key: str) -> None:
    assert config_problems(ARMS[key], RUN1, _preflight(key), sections=("train_config",)) == []


def test_run_1_itself_has_the_expected_baseline() -> None:
    """The fields the protocol says the arms change really were the Run 1 defaults."""
    train = RUN1["train_config"]
    assert (train["aug_config"], train["augmentation_backend"], train["seed"]) == (
        None,
        "cpu",
        None,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("lr", 2e-4), ("epochs", 40), ("scale_jitter", False), ("early_stopping_patience", 5)],
)
def test_any_other_drift_voids_the_run(field: str, value: object) -> None:
    saved = _preflight("2A")
    saved["train_config"][field] = value
    problems = config_problems(ARMS["2A"], RUN1, saved, sections=("train_config",))
    assert [p.split(":")[0] for p in problems] == [f"train_config.{field}"]


def test_wrong_treatment_values_void_the_run() -> None:
    saved = _preflight("2A")
    saved["train_config"]["seed"] = 1
    saved["train_config"]["aug_config"] = None
    problems = config_problems(ARMS["2A"], RUN1, saved, sections=("train_config",))
    assert sorted(p.split(":")[0] for p in problems) == [
        "train_config.aug_config",
        "train_config.seed",
    ]


def test_control_config_is_not_accepted_as_treatment() -> None:
    saved = _preflight("2C")
    assert config_problems(ARMS["2A"], RUN1, saved, sections=("train_config",))


def test_model_section_is_compared_and_weights_may_move_directory() -> None:
    saved = copy.deepcopy(RUN1)
    saved["train_config"] = _preflight("2A")["train_config"]
    assert config_problems(ARMS["2A"], RUN1, saved) == []

    saved["model_config"]["pretrain_weights"] = "/input/models/rf-detr-small.pth"
    assert config_problems(ARMS["2A"], RUN1, saved) == []

    saved["model_config"]["pretrain_weights"] = "/root/.roboflow/models/rf-detr-base.pth"
    saved["model_config"]["resolution"] = 560
    problems = config_problems(ARMS["2A"], RUN1, saved)
    assert sorted(p.split(":")[0] for p in problems) == [
        "model_config.pretrain_weights",
        "model_config.resolution",
    ]
