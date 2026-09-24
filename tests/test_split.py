"""Invariants of the committed split, checked from data/manifest.csv alone (no images needed)."""

from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from detector.dataset import CLASSES
from detector.dataset.manifest import Record, exclusion_reason, read_manifest
from detector.dataset.split import (
    NEAR_DUP_BITS,
    assign_splits,
    cross_split_near_duplicates,
    sessionize,
)

MANIFEST = Path(__file__).resolve().parents[1] / "data" / "manifest.csv"


@pytest.fixture(scope="module")
def records() -> list[Record]:
    return read_manifest(MANIFEST)


def test_inventory_matches_the_pinned_export(records: list[Record]) -> None:
    assert len(records) == 2251
    assert len({r.file for r in records}) == 2251
    assert len({r.sha256 for r in records}) == 2251  # no exact duplicate files
    assert Counter(r.published_split for r in records) == {"train": 1797, "valid": 232, "test": 222}
    assert sum(sum(r.box_counts.values()) for r in records) == 11453
    assert {c for r in records for c in r.box_counts} == set(CLASSES)


def test_exclusions_are_the_audited_ones(records: list[Record]) -> None:
    assert Counter(r.exclusion for r in records if r.exclusion) == {
        "third_party_video": 225,
        "third_party_press": 25,
        "unlabelled": 3,
        "double_labelled": 2,
    }
    for r in records:
        assert (r.split == "excluded") == bool(r.exclusion)
    # The filename rules agree with the recorded reason (double-labelling needs the boxes, so
    # it is checked against the export in test_local_export.py).
    for r in records:
        if r.exclusion != "double_labelled":
            anns = [(c, [k * 10, 0, 1, 1]) for k, c in enumerate(r.box_counts)]
            assert exclusion_reason(r.file, anns) == r.exclusion


def test_split_sizes(records: list[Record]) -> None:
    assert Counter(r.split for r in records) == {
        "train": 1263,
        "val": 331,
        "test": 402,
        "excluded": 255,
    }


def test_every_class_is_well_covered_in_every_split(records: list[Record]) -> None:
    for split, floor in (("train", 100), ("val", 40), ("test", 40)):
        images = Counter(c for r in records if r.split == split for c in r.box_counts)
        for c in CLASSES:
            assert images[c] >= floor, f"{c} has {images[c]} {split} images"


def test_test_split_is_whole_capture_dates(records: list[Record]) -> None:
    test_dates = {r.captured_at[:10] for r in records if r.split == "test"}
    others = {r.captured_at[:10] for r in records if r.split in ("train", "val")}
    assert test_dates == {"2025-05-04"}
    assert not test_dates & others


def test_no_session_spans_two_splits(records: list[Record]) -> None:
    split_of_session: dict[int, set[str]] = {}
    for r in records:
        if r.session is not None:
            split_of_session.setdefault(r.session, set()).add(r.split)
    assert all(len(s) == 1 for s in split_of_session.values())


def test_no_near_duplicates_cross_splits(records: list[Record]) -> None:
    assert cross_split_near_duplicates(records, max_bits=NEAR_DUP_BITS) == 0


def test_split_is_deterministic(records: list[Record]) -> None:
    fresh = deepcopy(records)
    for r in fresh:
        r.split, r.session = "", None
    assign_splits(fresh)
    assert [(r.split, r.session) for r in fresh] == [(r.split, r.session) for r in records]


def _rec(ts: str, bit: int) -> Record:
    h = np.zeros(256, dtype=np.uint8)
    h[bit * 40 : bit * 40 + 40] = 1  # each ``bit`` value is 80 bits away from the others
    return Record("f", "train", 1, 1, ts, "25kg:1", "", np.packbits(h).tobytes().hex())


def test_sessionize_breaks_on_time_gaps_and_merges_near_duplicates() -> None:
    from detector.dataset.split import hamming_matrix

    recs = [
        _rec("2025-05-01T10:00:00", 0),
        _rec("2025-05-01T10:03:00", 1),  # same session: 3 min gap
        _rec("2025-05-01T11:00:00", 2),  # new session: 57 min gap
        _rec("2025-05-02T09:00:00", 2),  # new day, but a near-duplicate of the one above
        _rec("2025-05-03T09:00:00", 3),
    ]
    sessions = sessionize(recs, hamming_matrix(recs))
    assert sessions[0] == sessions[1]
    assert sessions[2] == sessions[3]
    assert len({sessions[0], sessions[2], sessions[4]}) == 3
    assert sessions[0] == 1  # numbered chronologically
