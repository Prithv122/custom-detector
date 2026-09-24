"""Sorts a phone-dump folder into data/raw/gym_a/<load_id>/ by slate.

The capture protocol says "slate first: before each load, photograph its card" — so a
burst of photos taken close together in time is one load. This script can't read the
slate card itself (that needs eyes, not just timestamps), so it groups photos by a gap
in capture time and asks which load each burst belongs to; the grouping logic
(`group_by_gap`) is the only part worth unit-testing without real files.
"""

from __future__ import annotations

import csv
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic"}

# Matches common phone filename timestamps, e.g. IMG_20260924_101503.jpg
_FILENAME_TS_RE = re.compile(r"(\d{4})(\d{2})(\d{2})[_-]?(\d{2})(\d{2})(\d{2})")


def _timestamp_for(path: Path) -> datetime:
    match = _FILENAME_TS_RE.search(path.stem)
    if match:
        year, month, day, hour, minute, second = (int(g) for g in match.groups())
        try:
            return datetime(year, month, day, hour, minute, second)
        except ValueError:
            pass  # fall through to mtime if the filename digits aren't a real date
    return datetime.fromtimestamp(path.stat().st_mtime)


def group_by_gap(timestamps: list[datetime], gap_seconds: float) -> list[list[int]]:
    """Group timestamp indices into bursts; a new burst starts after a gap > gap_seconds.

    Pure function so the bursting heuristic is testable without real photo files.
    """
    if not timestamps:
        return []
    order = sorted(range(len(timestamps)), key=lambda i: timestamps[i])
    groups: list[list[int]] = [[order[0]]]
    for prev_idx, idx in pairwise(order):
        gap = (timestamps[idx] - timestamps[prev_idx]).total_seconds()
        if gap > gap_seconds:
            groups.append([])
        groups[-1].append(idx)
    return groups


@dataclass(frozen=True)
class SortedPhoto:
    source: Path
    dest: Path
    load_id: str


def list_source_photos(source_dir: Path) -> list[Path]:
    return sorted(
        p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def default_ask_load_id(burst_num: int, count: int, first_timestamp: datetime) -> str:
    prompt = (
        f"Burst {burst_num}: {count} photo(s) starting {first_timestamp:%Y-%m-%d %H:%M:%S}. "
        "Slate/load id (e.g. L07 or LOOSE), or blank to skip: "
    )
    return input(prompt)


def sort_by_slate(
    source_dir: Path,
    dest_dir: Path,
    gap_seconds: float = 120.0,
    ask_load_id: Callable[[int, int, datetime], str] = default_ask_load_id,
) -> list[SortedPhoto]:
    """Copy (never move) photos from source_dir into dest_dir/<load_id>/, burst by burst."""
    photos = list_source_photos(source_dir)
    if not photos:
        return []

    timestamps = [_timestamp_for(p) for p in photos]
    groups = group_by_gap(timestamps, gap_seconds)

    moved: list[SortedPhoto] = []
    manifest_rows: list[tuple[str, str, str, str]] = []
    for burst_num, group in enumerate(groups, start=1):
        first_idx = group[0]
        load_id = ask_load_id(burst_num, len(group), timestamps[first_idx]).strip().upper()
        if not load_id:
            continue

        target_dir = dest_dir / load_id
        target_dir.mkdir(parents=True, exist_ok=True)
        for seq, idx in enumerate(sorted(group, key=lambda i: timestamps[i]), start=1):
            src = photos[idx]
            dest = target_dir / f"{load_id}_{seq:02d}{src.suffix.lower()}"
            shutil.copy2(src, dest)
            moved.append(SortedPhoto(src, dest, load_id))
            manifest_rows.append((str(src), str(dest), load_id, timestamps[idx].isoformat()))

    if manifest_rows:
        manifest_path = dest_dir / "manifest.csv"
        write_header = not manifest_path.exists()
        dest_dir.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow(["source", "dest", "load_id", "timestamp"])
            writer.writerows(manifest_rows)

    return moved
