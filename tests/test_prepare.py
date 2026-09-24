import json
from pathlib import Path

import pytest

from detector.dataset.manifest import build_manifest
from detector.dataset.prepare import materialize


def test_materialize_writes_only_kept_images_under_the_manifest_split(
    export: Path, tmp_path: Path
) -> None:
    records = build_manifest(export)
    wanted = {"IMG_20250501_120000.jpg": "test", "IMG_20250502_090500.jpg": "train"}
    for r in records:
        r.split = wanted.get(r.file, "excluded")

    counts = materialize(export, records, tmp_path / "out")

    assert counts == {"train": 1, "val": 0, "test": 1}
    test = json.loads((tmp_path / "out" / "test" / "_annotations.coco.json").read_text())
    assert [im["extra"]["name"] for im in test["images"]] == ["IMG_20250501_120000.jpg"]
    assert [im["id"] for im in test["images"]] == [0]
    assert {a["image_id"] for a in test["annotations"]} == {0}
    assert len(test["annotations"]) == 2
    assert test["categories"][1]["name"] == "25kg"
    assert (tmp_path / "out" / "test" / test["images"][0]["file_name"]).exists()
    # val uses Roboflow's folder name so RF-DETR reads it directly
    assert (tmp_path / "out" / "valid" / "_annotations.coco.json").exists()
    assert len(list((tmp_path / "out" / "train").glob("*.jpg"))) == 1


def test_materialize_refuses_images_missing_from_the_manifest(export: Path, tmp_path: Path) -> None:
    records = build_manifest(export)[1:]
    with pytest.raises(KeyError, match="not in the manifest"):
        materialize(export, records, tmp_path / "out")
