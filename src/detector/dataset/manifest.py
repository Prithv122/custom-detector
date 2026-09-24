"""Build the per-image manifest from a Roboflow COCO export and apply the exclusion rules."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path

import imagehash
from PIL import Image

EXPORT_SPLITS = ("train", "valid", "test")
PHONE_TIMESTAMP = re.compile(r"(?:IMG|VID|PXL)_(\d{8})_(\d{6})")
CONFLICT_IOU = 0.9

# Exclusion reasons, in precedence order. Provenance comes first: an image we can't publish is
# excluded for that reason even if its labels are also broken.
THIRD_PARTY_VIDEO = "third_party_video"
THIRD_PARTY_PRESS = "third_party_press"
UNLABELLED = "unlabelled"
DOUBLE_LABELLED = "double_labelled"


@dataclass
class Record:
    """One image in the export. ``split`` is filled in by :mod:`detector.dataset.split`."""

    file: str
    published_split: str
    width: int
    height: int
    captured_at: str
    boxes: str
    sha256: str
    phash256: str
    exclusion: str = ""
    session: int | None = None
    split: str = ""
    _box_counts: Counter = field(default_factory=Counter, repr=False, compare=False)

    @property
    def box_counts(self) -> Counter:
        if not self._box_counts and self.boxes:
            self._box_counts = Counter(
                {c: int(n) for c, n in (p.split(":") for p in self.boxes.split(";"))}
            )
        return self._box_counts


CSV_FIELDS = [f.name for f in fields(Record) if not f.name.startswith("_")]


def parse_captured_at(name: str) -> str:
    """ISO timestamp from a phone-camera filename (``IMG_20251106_130044``), else ``""``."""
    m = PHONE_TIMESTAMP.search(name)
    if not m:
        return ""
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").isoformat()


def _iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def has_conflicting_labels(anns: list[tuple[str, list[float]]]) -> bool:
    """True if one object carries two different class labels (boxes of different class, IoU>0.9)."""
    for i, (ca, ba) in enumerate(anns):
        for cb, bb in anns[i + 1 :]:
            if ca != cb and _iou(ba, bb) > CONFLICT_IOU:
                return True
    return False


def exclusion_reason(name: str, anns: list[tuple[str, list[float]]]) -> str:
    """Why an image is excluded, or ``""`` if it is kept.

    The filename rules were confirmed by spot-checking a sample of each family: ``ZE_*`` files are
    crops of competition video, and files without a phone-camera timestamp are press photos from
    international competitions. Neither can be the uploader's own photograph.
    """
    if name.startswith("ZE_"):
        return THIRD_PARTY_VIDEO
    if not parse_captured_at(name):
        return THIRD_PARTY_PRESS
    if not anns:
        return UNLABELLED
    if has_conflicting_labels(anns):
        return DOUBLE_LABELLED
    return ""


def _format_boxes(anns: list[tuple[str, list[float]]]) -> str:
    counts = Counter(c for c, _ in anns)
    return ";".join(f"{c}:{counts[c]}" for c in sorted(counts))


def build_manifest(export_dir: Path) -> list[Record]:
    """Read every image and annotation in a Roboflow COCO export into manifest records."""
    records: list[Record] = []
    for split in EXPORT_SPLITS:
        coco = json.loads((export_dir / split / "_annotations.coco.json").read_text())
        names = {c["id"]: c["name"] for c in coco["categories"]}
        anns_by_image: dict[int, list[tuple[str, list[float]]]] = {}
        for a in coco["annotations"]:
            anns_by_image.setdefault(a["image_id"], []).append((names[a["category_id"]], a["bbox"]))
        for im in coco["images"]:
            path = export_dir / split / im["file_name"]
            original = im.get("extra", {}).get("name", im["file_name"])
            anns = anns_by_image.get(im["id"], [])
            with Image.open(path) as img:
                phash = imagehash.phash(img.convert("RGB"), hash_size=16)
            records.append(
                Record(
                    file=original,
                    published_split=split,
                    width=im["width"],
                    height=im["height"],
                    captured_at=parse_captured_at(original),
                    boxes=_format_boxes(anns),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    phash256=str(phash),
                    exclusion=exclusion_reason(original, anns),
                )
            )
    return records


def write_manifest(records: list[Record], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        w.writeheader()
        for r in records:
            row = {k: getattr(r, k) for k in CSV_FIELDS}
            row["session"] = "" if r.session is None else r.session
            w.writerow(row)


def read_manifest(path: Path) -> list[Record]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        Record(
            file=r["file"],
            published_split=r["published_split"],
            width=int(r["width"]),
            height=int(r["height"]),
            captured_at=r["captured_at"],
            boxes=r["boxes"],
            sha256=r["sha256"],
            phash256=r["phash256"],
            exclusion=r["exclusion"],
            session=int(r["session"]) if r["session"] else None,
            split=r["split"],
        )
        for r in rows
    ]
