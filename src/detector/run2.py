"""Run 2 switch: the two training arms (2C control, 2A colour cast) and the config check.

Implements "Run 2 - colour-cast augmentation" in NOTES.md. The two arms share one commit and
differ only in ``aug_config``; the notebook picks one with a single ``RUN`` value. Keeping the
definitions here, not in the notebook, lets a local test prove they build through ``rfdetr``
and that every other training field still equals Run 1's saved config.

``rfdetr`` builds user augmentation leniently: a transform that fails to build is logged and
skipped. A typo in the colour transform would therefore train a second control and look fine,
so ``build_transforms`` builds strictly and checks the result by name.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

ALBUMENTATIONS_VERSION = "2.0.8"
SEED = 20261001
BACKEND = "albumentations"

FLIP = {"HorizontalFlip": {"p": 0.5}}
PLANCKIAN = {
    "PlanckianJitter": {
        "mode": "cied",
        "temperature_limit": (4000, 15000),
        "sampling_method": "uniform",
        "p": 0.5,
    }
}


@dataclass(frozen=True)
class Arm:
    key: str
    label: str
    results_dir: str
    aug_config: dict[str, dict[str, Any]]

    @property
    def transform_names(self) -> list[str]:
        return list(self.aug_config)

    def train_kwargs(self) -> dict[str, Any]:
        """What the arm adds to the notebook's ``TRAIN`` dict (a copy, so nothing is shared)."""
        return {
            "aug_config": copy.deepcopy(self.aug_config),
            "augmentation_backend": BACKEND,
            "seed": SEED,
        }


ARMS: dict[str, Arm] = {
    "2C": Arm("2C", "control: flip only, Albumentations path", "run2c", {**FLIP}),
    "2A": Arm("2A", "treatment: flip + PlanckianJitter", "run2a", {**FLIP, **PLANCKIAN}),
}


def get_arm(key: str) -> Arm:
    if key not in ARMS:
        raise ValueError(f"RUN must be one of {sorted(ARMS)}, got {key!r}")
    return ARMS[key]


def build_transforms(arm: Arm) -> list[Any]:
    """Build the arm's augmentation through ``rfdetr`` and fail if anything was dropped.

    ``strict=True`` raises on a transform that cannot be built, and the name check catches a
    list that is shorter or reordered for any other reason.
    """
    from rfdetr.datasets.transforms import AlbumentationsWrapper

    wrappers = AlbumentationsWrapper.from_config(copy.deepcopy(arm.aug_config), strict=True)
    built = [_inner_transform(w).__class__.__name__ for w in wrappers]
    if built != arm.transform_names:
        raise RuntimeError(f"arm {arm.key}: built {built}, expected {arm.transform_names}")
    return wrappers


def _inner_transform(wrapper: Any) -> Any:
    return wrapper.transform.transforms[0]


# --- config check ---------------------------------------------------------------------------

# Fields allowed to differ from Run 1 inside ``train_config`` (NOTES.md, Run 2, section 3).
ALLOWED_TRAIN_DIFFS = frozenset(
    {"aug_config", "augmentation_backend", "seed", "dataset_dir", "output_dir"}
)
# The weights file must be the same, but it may sit in a different cache directory.
PATH_LIKE = ("model_config.pretrain_weights",)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    return {prefix: value}


def _jsonable(value: Any) -> Any:
    """Round-trip through JSON, as the saved file was (tuples become lists)."""
    return json.loads(json.dumps(value))


def _is_allowed(field: str) -> bool:
    section, _, rest = field.partition(".")
    return section == "train_config" and rest.split(".")[0] in ALLOWED_TRAIN_DIFFS


def _basename(path: Any) -> Any:
    return str(path).replace("\\", "/").rsplit("/", 1)[-1] if isinstance(path, str) else path


def config_problems(
    arm: Arm,
    run1: dict[str, Any],
    saved: dict[str, Any],
    sections: tuple[str, ...] = ("train_config", "model_config", "class_names", "num_classes"),
) -> list[str]:
    """Reasons ``saved`` voids the run; an empty list means the config check passes.

    Compares ``saved`` with Run 1's ``training_config.json`` field by field, then checks that
    the three allowed treatment fields hold the arm's own values.
    """
    problems: list[str] = []
    a = _flatten({s: run1[s] for s in sections if s in run1})
    b = _flatten({s: saved[s] for s in sections if s in saved})
    for field in sorted(a.keys() | b.keys()):
        if _is_allowed(field):
            continue
        if field in PATH_LIKE:
            same = _basename(a.get(field)) == _basename(b.get(field))
        else:
            same = field in a and field in b and a[field] == b[field]
        if not same:
            problems.append(
                f"{field}: run 1 {a.get(field, '<absent>')!r}, saved {b.get(field, '<absent>')!r}"
            )

    train = saved.get("train_config", {})
    expected = _jsonable(arm.train_kwargs())
    for field, want in expected.items():
        if train.get(field) != want:
            problems.append(
                f"train_config.{field}: arm {arm.key} needs {want!r}, saved {train.get(field)!r}"
            )
    return problems
