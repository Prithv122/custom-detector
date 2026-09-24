"""Validates data/loads.csv against the rules in data/CAPTURE_PROTOCOL.md.

loads.csv is the independent ground truth — recorded from what was physically on the bar,
never derived from box labels. This only checks internal consistency (arithmetic, allowed
plate classes, known load ids); it can't and shouldn't check that the recording itself was
honest.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from detector.protocol import LOOSE_LOAD_ID, VALID_PLATE_CLASSES_KG, parse_load_schedule

REQUIRED_COLUMNS = (
    "load_id",
    "gym",
    "date",
    "bar_kg",
    "collar_kg_each",
    "side_plates_kg",
    "total_kg",
    "photos",
    "notes",
)

_EXPECTED_PHOTOS_PER_LOAD = 6


@dataclass(frozen=True)
class Issue:
    row: int | None  # None for file-level issues (header, missing loads)
    level: str  # "error" or "warning"
    message: str


@dataclass(frozen=True)
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)
    loads_seen: set[str] = field(default_factory=set)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors


def _parse_plates(raw: str) -> list[float]:
    raw = raw.strip()
    if not raw:
        return []
    return [float(part.strip()) for part in raw.split("+")]


def validate_loads_csv(csv_path: Path, protocol_path: Path) -> ValidationReport:
    schedule = parse_load_schedule(protocol_path)
    issues: list[Issue] = []
    loads_seen: set[str] = set()

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or list(reader.fieldnames) != list(REQUIRED_COLUMNS):
            issues.append(
                Issue(
                    None,
                    "error",
                    f"header mismatch: expected {list(REQUIRED_COLUMNS)}, got {reader.fieldnames}",
                )
            )
            return ValidationReport(issues, loads_seen)

        for row_num, row in enumerate(reader, start=2):  # header is line 1
            load_id = row["load_id"].strip()
            if not load_id:
                issues.append(Issue(row_num, "error", "empty load_id"))
                continue
            loads_seen.add(load_id)

            spec = schedule.get(load_id)
            if spec is None:
                issues.append(
                    Issue(row_num, "error", f"{load_id} is not in the capture protocol schedule")
                )
                continue

            try:
                bar_kg = float(row["bar_kg"])
                collar_kg_each = float(row["collar_kg_each"] or 0)
                total_kg = float(row["total_kg"])
            except ValueError as exc:
                issues.append(
                    Issue(row_num, "error", f"{load_id}: non-numeric bar/collar/total ({exc})")
                )
                continue

            try:
                plates = _parse_plates(row["side_plates_kg"])
            except ValueError:
                issues.append(
                    Issue(
                        row_num,
                        "error",
                        f"{load_id}: side_plates_kg must be '+'-separated numbers",
                    )
                )
                continue

            if load_id != LOOSE_LOAD_ID:
                bad_classes = sorted({p for p in plates if p not in VALID_PLATE_CLASSES_KG})
                if bad_classes:
                    issues.append(
                        Issue(
                            row_num,
                            "error",
                            f"{load_id}: plate class(es) {bad_classes} outside the six allowed "
                            "classes",
                        )
                    )

                expected_total = bar_kg + 2 * sum(plates) + 2 * collar_kg_each
                if abs(expected_total - total_kg) > 0.01:
                    issues.append(
                        Issue(
                            row_num,
                            "error",
                            f"{load_id}: total_kg={total_kg} doesn't match "
                            f"bar+plates+collars={expected_total}",
                        )
                    )

                if tuple(sorted(plates)) != tuple(sorted(spec.per_side_kg)):
                    issues.append(
                        Issue(
                            row_num,
                            "warning",
                            f"{load_id}: recorded plates {sorted(plates)} differ from the "
                            f"schedule {sorted(spec.per_side_kg)} (fine if substituted, per "
                            "protocol — just confirm it was intentional)",
                        )
                    )

            photos = row["photos"].strip()
            if (
                photos.isdigit()
                and load_id != LOOSE_LOAD_ID
                and int(photos) != _EXPECTED_PHOTOS_PER_LOAD
            ):
                issues.append(
                    Issue(
                        row_num,
                        "warning",
                        f"{load_id}: {photos} photo(s) recorded, protocol calls for "
                        f"{_EXPECTED_PHOTOS_PER_LOAD}",
                    )
                )

    missing = set(schedule) - loads_seen
    if missing:
        issues.append(
            Issue(None, "warning", f"{len(missing)} load(s) not yet recorded: {sorted(missing)}")
        )

    return ValidationReport(issues, loads_seen)
