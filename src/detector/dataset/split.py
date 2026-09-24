"""Leakage-free train/val/test assignment by capture date and session.

The published Roboflow split leaks: photos of one rig taken seconds apart sit in different
splits. Here the unit of assignment is a group of photos, never a single photo:

1. **Sessions.** Sort kept images by capture time; a gap of more than ``SESSION_GAP_MIN``
   minutes starts a new session. Any two sessions linked by a near-duplicate pair
   (256-bit pHash distance <= ``NEAR_DUP_BITS``) are merged.
2. **Test = whole capture dates.** Among sets of dates holding 15-25% of kept images, where
   every class has >= 15 test images and >= 100 images left for training, take the set
   closest to 20%. Test therefore measures a day the model has never seen.
3. **Val = whole sessions** from the remaining dates: 2,000 seeded permutations, each filled
   greedily to 12-17% of kept images; keep the one whose rarest class has the most images.
4. **Train** = everything else.

Every choice uses dataset support only (dates, sessions, class counts), never model results.
"""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

from detector.dataset import CLASSES
from detector.dataset.manifest import Record

SESSION_GAP_MIN = 5
NEAR_DUP_BITS = 30
TEST_FRACTION = (0.15, 0.25)
TEST_TARGET = 0.20
TEST_MIN_IMAGES_PER_CLASS = 15
TRAIN_MIN_IMAGES_PER_CLASS = 100
VAL_FRACTION = (0.12, 0.17)
VAL_TRIALS = 2000
SEED = 26


def hash_bits(records: list[Record]) -> np.ndarray:
    """(n, 256) uint8 bit matrix from the hex pHashes."""
    raw = np.array([bytes.fromhex(r.phash256) for r in records], dtype="S32")
    return np.unpackbits(
        np.frombuffer(raw.tobytes(), dtype=np.uint8).reshape(len(records), 32), axis=1
    )


def hamming_matrix(records: list[Record], chunk: int = 256) -> np.ndarray:
    bits = hash_bits(records)
    out = np.zeros((len(bits), len(bits)), dtype=np.int32)
    for s in range(0, len(bits), chunk):
        out[s : s + chunk] = (bits[s : s + chunk, None, :] != bits[None, :, :]).sum(-1)
    return out


def sessionize(records: list[Record], dist: np.ndarray) -> list[int]:
    """Session id per record (records must all have ``captured_at``), numbered chronologically."""
    order = sorted(range(len(records)), key=lambda i: records[i].captured_at)
    raw, s, prev = {}, 0, None
    for i in order:
        t = datetime.fromisoformat(records[i].captured_at)
        if prev and (t - prev).total_seconds() > SESSION_GAP_MIN * 60:
            s += 1
        raw[i] = s
        prev = t

    parent = {x: x for x in set(raw.values())}

    def find(x: int) -> int:
        while parent[x] != x:
            x = parent[x]
        return x

    a, b = np.where(np.triu(dist <= NEAR_DUP_BITS, 1))
    for x, y in zip(a.tolist(), b.tolist(), strict=True):
        rx, ry = find(raw[x]), find(raw[y])
        if rx != ry:
            parent[rx] = ry

    root = [find(raw[i]) for i in range(len(records))]
    first_seen = sorted(
        set(root),
        key=lambda g: min(records[i].captured_at for i in range(len(records)) if root[i] == g),
    )
    number = {g: k + 1 for k, g in enumerate(first_seen)}
    return [number[g] for g in root]


def _images_per_class(records: list[Record], idx: list[int]) -> Counter:
    c: Counter = Counter()
    for i in idx:
        c.update(records[i].box_counts.keys())
    return c


def choose_test_dates(records: list[Record]) -> tuple[str, ...]:
    by_date: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_date[r.captured_at[:10]].append(i)
    dates = sorted(by_date)
    n = len(records)
    candidates = []
    for k in range(1, len(dates)):
        for combo in itertools.combinations(dates, k):
            idx = [i for d in combo for i in by_date[d]]
            frac = len(idx) / n
            chosen = set(idx)
            test_im = _images_per_class(records, idx)
            rest_im = _images_per_class(records, [i for i in range(n) if i not in chosen])
            if (
                TEST_FRACTION[0] <= frac <= TEST_FRACTION[1]
                and all(test_im[c] >= TEST_MIN_IMAGES_PER_CLASS for c in CLASSES)
                and all(rest_im[c] >= TRAIN_MIN_IMAGES_PER_CLASS for c in CLASSES)
            ):
                candidates.append((abs(frac - TEST_TARGET), combo))
    if not candidates:
        raise ValueError("no set of capture dates satisfies the test-split rule")
    return min(candidates)[1]


def choose_val_sessions(records: list[Record], sessions: list[int], pool: list[int]) -> list[int]:
    """Indices (into ``records``) of the validation images, chosen from ``pool``."""
    n = len(records)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in pool:
        groups[sessions[i]].append(i)
    keys = sorted(groups, key=lambda g: min(records[i].captured_at for i in groups[g]))
    lo, hi = VAL_FRACTION[0] * n, VAL_FRACTION[1] * n
    rng = np.random.default_rng(SEED)
    best: tuple[int, list[int]] | None = None
    for _ in range(VAL_TRIALS):
        val: list[int] = []
        for p in rng.permutation(len(keys)):
            if len(val) + len(groups[keys[p]]) <= hi:
                val += groups[keys[p]]
            if len(val) >= lo:
                break
        if not lo <= len(val) <= hi:
            continue
        per_class = _images_per_class(records, val)
        score = min(per_class[c] for c in CLASSES)
        if best is None or score > best[0]:
            best = (score, val)
    if best is None:
        raise ValueError("no combination of sessions satisfies the validation-split rule")
    return best[1]


def assign_splits(records: list[Record]) -> None:
    """Fill ``session`` and ``split`` on every record in place. Excluded records get
    ``split="excluded"`` and no session."""
    kept = [r for r in records if not r.exclusion]
    for r in records:
        if r.exclusion:
            r.split, r.session = "excluded", None

    dist = hamming_matrix(kept)
    sessions = sessionize(kept, dist)
    test_dates = set(choose_test_dates(kept))
    pool = [i for i, r in enumerate(kept) if r.captured_at[:10] not in test_dates]
    val = set(choose_val_sessions(kept, sessions, pool))

    for i, r in enumerate(kept):
        r.session = sessions[i]
        if r.captured_at[:10] in test_dates:
            r.split = "test"
        elif i in val:
            r.split = "val"
        else:
            r.split = "train"


def cross_split_near_duplicates(records: list[Record], max_bits: int = NEAR_DUP_BITS) -> int:
    """Number of image pairs in different splits whose pHashes are within ``max_bits``."""
    kept = [r for r in records if r.split in ("train", "val", "test")]
    dist = hamming_matrix(kept)
    labels = np.array([r.split for r in kept])
    close = np.triu(dist <= max_bits, 1)
    diff = labels[:, None] != labels[None, :]
    return int((close & diff).sum())
