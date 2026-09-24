"""Download the pinned Roboflow export and write the cleaned, re-split COCO dataset."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from detector.dataset.manifest import EXPORT_SPLITS, Record

WORKSPACE = "projektciezary"
PROJECT = "weightlifting-plates"
VERSION = 10

# RF-DETR reads a Roboflow-style COCO layout: train/, valid/, test/, each with
# _annotations.coco.json next to the images.
OUTPUT_DIRS = {"train": "train", "val": "valid", "test": "test"}


def download(dest: Path, api_key: str) -> Path:
    """Fetch the pinned export (COCO JSON) into ``dest``. Needs a free Roboflow API key."""
    from roboflow import Roboflow

    version = Roboflow(api_key=api_key).workspace(WORKSPACE).project(PROJECT).version(VERSION)
    version.download("coco", location=str(dest), overwrite=False)
    return dest


def materialize(export_dir: Path, records: list[Record], out_dir: Path) -> dict[str, int]:
    """Copy kept images into ``out_dir`` under the manifest split, with one COCO file per split.

    Categories are copied unchanged from the export so class ids stay stable. Returns the number
    of images written per split.
    """
    split_of = {r.file: r.split for r in records}
    categories = None
    images: dict[str, list[dict]] = {s: [] for s in OUTPUT_DIRS}
    annotations: dict[str, list[dict]] = {s: [] for s in OUTPUT_DIRS}
    next_ann = dict.fromkeys(OUTPUT_DIRS, 0)

    for d in OUTPUT_DIRS.values():
        (out_dir / d).mkdir(parents=True, exist_ok=True)

    for src_split in EXPORT_SPLITS:
        coco = json.loads((export_dir / src_split / "_annotations.coco.json").read_text())
        if categories is None:
            categories = coco["categories"]
        elif coco["categories"] != categories:
            raise ValueError(f"categories in {src_split} differ from the other splits")
        anns_by_image: dict[int, list[dict]] = {}
        for a in coco["annotations"]:
            anns_by_image.setdefault(a["image_id"], []).append(a)

        for im in coco["images"]:
            original = im.get("extra", {}).get("name", im["file_name"])
            target = split_of.get(original)
            if target is None:
                raise KeyError(f"{original} is in the export but not in the manifest")
            if target not in OUTPUT_DIRS:
                continue
            new_id = len(images[target])
            shutil.copy2(export_dir / src_split / im["file_name"], out_dir / OUTPUT_DIRS[target])
            images[target].append({**im, "id": new_id})
            for a in anns_by_image.get(im["id"], []):
                annotations[target].append({**a, "id": next_ann[target], "image_id": new_id})
                next_ann[target] += 1

    for s, d in OUTPUT_DIRS.items():
        coco_out = {"categories": categories, "images": images[s], "annotations": annotations[s]}
        (out_dir / d / "_annotations.coco.json").write_text(json.dumps(coco_out))
    return {s: len(v) for s, v in images.items()}
