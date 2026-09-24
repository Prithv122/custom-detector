"""Rebuild the manifest from the real export and compare it with the committed one.

Needs the downloaded export (``custom-detector download``), so it is skipped in CI.
"""

from pathlib import Path

import pytest

from detector.dataset.manifest import build_manifest, read_manifest
from detector.dataset.split import assign_splits

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "data" / "raw" / "projektciezary_v10_coco"

pytestmark = pytest.mark.skipif(not EXPORT.exists(), reason="Roboflow export not downloaded")


def test_committed_manifest_is_reproducible_from_the_export() -> None:
    rebuilt = build_manifest(EXPORT)
    assign_splits(rebuilt)
    assert rebuilt == read_manifest(ROOT / "data" / "manifest.csv")
