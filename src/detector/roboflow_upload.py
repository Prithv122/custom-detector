"""Uploads data/raw/gym_a/<load_id>/*.jpg to Roboflow with the pre-fixed split.

The split (18 train / 3 val / 3 test loads) is fixed in data/CAPTURE_PROTOCOL.md before any
training, specifically so Roboflow's own random splitter — which would leak near-duplicate
photos of the same load across train/val/test — never gets a say.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from detector.protocol import LOOSE_LOAD_ID, LoadSpec, parse_load_schedule
from detector.sort_photos import IMAGE_EXTENSIONS


class UploadableProject(Protocol):
    def upload(self, image_path: str, split: str, tag_names: list[str]) -> object: ...


@dataclass
class UploadResult:
    uploaded: dict[str, int] = field(default_factory=dict)  # split -> count
    failures: list[tuple[Path, str]] = field(default_factory=list)


def resolve_split(load_id: str, schedule: dict[str, LoadSpec]) -> str:
    if load_id == LOOSE_LOAD_ID:
        return "train"
    spec = schedule.get(load_id)
    if spec is None:
        raise KeyError(f"{load_id} is not in the capture protocol schedule")
    return spec.split


def upload_gym_a(
    source_dir: Path,
    protocol_path: Path,
    project: UploadableProject,
) -> UploadResult:
    schedule = parse_load_schedule(protocol_path)
    result = UploadResult()

    for load_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
        load_id = load_dir.name
        try:
            split = resolve_split(load_id, schedule)
        except KeyError as exc:
            result.failures.append((load_dir, str(exc)))
            continue

        images = sorted(p for p in load_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        for image_path in images:
            try:
                project.upload(image_path=str(image_path), split=split, tag_names=[load_id])
            except Exception as exc:  # one bad upload shouldn't abort the whole batch
                result.failures.append((image_path, str(exc)))
                continue
            result.uploaded[split] = result.uploaded.get(split, 0) + 1

    return result


def build_roboflow_project(api_key: str, workspace: str, project_slug: str) -> UploadableProject:
    from roboflow import Roboflow  # imported lazily so tests don't need network/credentials

    rf = Roboflow(api_key=api_key)
    return rf.workspace(workspace).project(project_slug)
