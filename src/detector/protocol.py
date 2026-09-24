"""Parses the Gym A load schedule out of data/CAPTURE_PROTOCOL.md.

The schedule (per-side plates, total, train/val/test split) is fixed there, before any
training — this module reads that table directly instead of duplicating it, so the two
can never drift apart.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

VALID_PLATE_CLASSES_KG: frozenset[float] = frozenset({25.0, 20.0, 15.0, 10.0, 5.0, 2.5})
LOOSE_LOAD_ID = "LOOSE"

_TABLE_ROW_RE = re.compile(
    r"^\|\s*(L\d{2})\s*\|\s*([^|]+?)\s*\|\s*([\d.]+)\s*\|\s*\**\s*(train|val|test)\s*\**\s*\|\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LoadSpec:
    load_id: str
    per_side_kg: tuple[float, ...]
    total_kg: float
    split: str


def _parse_per_side(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in raw.split("+"))


def parse_load_schedule(protocol_path: Path) -> dict[str, LoadSpec]:
    """Read the "Load schedule — Gym A" table into {load_id: LoadSpec}, plus LOOSE."""
    text = protocol_path.read_text(encoding="utf-8")
    schedule: dict[str, LoadSpec] = {}
    in_table = False
    for line in text.splitlines():
        if line.startswith("## Load schedule"):
            in_table = True
            continue
        if in_table and line.startswith("## "):
            break
        if not in_table:
            continue
        match = _TABLE_ROW_RE.match(line.strip())
        if not match:
            continue
        load_id, per_side_raw, total_raw, split = match.groups()
        schedule[load_id] = LoadSpec(
            load_id=load_id,
            per_side_kg=_parse_per_side(per_side_raw),
            total_kg=float(total_raw),
            split=split.lower(),
        )

    schedule[LOOSE_LOAD_ID] = LoadSpec(
        load_id=LOOSE_LOAD_ID, per_side_kg=(), total_kg=0.0, split="train"
    )
    return schedule
