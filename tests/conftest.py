"""Shared fixtures: a tiny synthetic Roboflow COCO export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

CATEGORIES = [
    {"id": 0, "name": "objects", "supercategory": "none"},
    {"id": 1, "name": "25kg", "supercategory": "objects"},
    {"id": 2, "name": "2.5kg", "supercategory": "objects"},
    {"id": 3, "name": "zacisk", "supercategory": "objects"},
]


def write_export(
    root: Path, spec: dict[str, list[tuple[str, list[tuple[int, list[float]]]]]]
) -> Path:
    """``spec`` maps split -> [(original_name, [(category_id, bbox), ...]), ...]."""
    for split, items in spec.items():
        d = root / split
        d.mkdir(parents=True)
        images, anns = [], []
        for i, (name, boxes) in enumerate(items):
            fname = f"{name.replace('.', '_')}.rf.{i:032x}.jpg"
            # Different shades so every synthetic file has a distinct SHA-256.
            Image.new("RGB", (64, 48), (i * 40 % 256, 90, 160)).save(d / fname)
            images.append(
                {"id": i, "file_name": fname, "width": 64, "height": 48, "extra": {"name": name}}
            )
            for cat, bbox in boxes:
                anns.append({"id": len(anns), "image_id": i, "category_id": cat, "bbox": bbox})
        coco = {"categories": CATEGORIES, "images": images, "annotations": anns}
        (d / "_annotations.coco.json").write_text(json.dumps(coco))
    return root


@pytest.fixture
def export(tmp_path: Path) -> Path:
    return write_export(
        tmp_path / "export",
        {
            "train": [
                ("IMG_20250501_120000.jpg", [(1, [1, 1, 10, 30]), (3, [12, 1, 5, 10])]),
                ("ZE_7.png", [(1, [1, 1, 10, 30])]),
            ],
            "valid": [
                ("IMG_20250501_120005.jpg", []),
                ("12.webp", [(2, [1, 1, 4, 20])]),
            ],
            "test": [
                ("IMG_20250502_090000.jpg", [(2, [5, 5, 4, 20]), (1, [5, 5, 4, 21])]),
                ("IMG_20250502_090500.jpg", [(2, [5, 5, 4, 20]), (3, [20, 5, 4, 10])]),
            ],
        },
    )
