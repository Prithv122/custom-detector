"""Console entry point: download the export, build the manifest, prepare the training set."""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from detector.dataset import CLASSES
from detector.dataset.manifest import build_manifest, read_manifest, write_manifest
from detector.dataset.prepare import download, materialize
from detector.dataset.split import assign_splits, cross_split_near_duplicates

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
EXPORT_DIR = DATA_DIR / "raw" / "projektciezary_v10_coco"
MANIFEST = DATA_DIR / "manifest.csv"
PREPARED_DIR = DATA_DIR / "processed" / "plates_v10_clean"


def _cmd_download(args: argparse.Namespace) -> int:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("ROBOFLOW_API_KEY is not set (see .env.example)", file=sys.stderr)
        return 1
    download(Path(args.dest), api_key)
    return 0


def _cmd_manifest(args: argparse.Namespace) -> int:
    records = build_manifest(Path(args.export))
    assign_splits(records)
    write_manifest(records, Path(args.out))
    print(f"Wrote {len(records)} records to {args.out}")
    return _summary(records)


def _cmd_prepare(args: argparse.Namespace) -> int:
    counts = materialize(Path(args.export), read_manifest(Path(args.manifest)), Path(args.out))
    for split, n in counts.items():
        print(f"{split}: {n} images")
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    return _summary(read_manifest(Path(args.manifest)))


def _summary(records: list) -> int:
    print("exclusions:", dict(Counter(r.exclusion for r in records if r.exclusion)))
    splits = ("train", "val", "test")
    print("| class | " + " | ".join(splits) + " |  (images / boxes)")
    for c in CLASSES:
        cells = []
        for s in splits:
            rs = [r for r in records if r.split == s]
            cells.append(
                f"{sum(c in r.box_counts for r in rs)} / {sum(r.box_counts[c] for r in rs)}"
            )
        print(f"| {c} | " + " | ".join(cells) + " |")
    print("images:", {s: sum(r.split == s for r in records) for s in splits})
    print("cross-split near-duplicate pairs:", cross_split_near_duplicates(records))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="custom-detector")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="Download the pinned Roboflow export (needs API key)")
    p.add_argument("--dest", default=str(EXPORT_DIR))
    p.set_defaults(func=_cmd_download)

    p = sub.add_parser("manifest", help="Rebuild data/manifest.csv from the export")
    p.add_argument("--export", default=str(EXPORT_DIR))
    p.add_argument("--out", default=str(MANIFEST))
    p.set_defaults(func=_cmd_manifest)

    p = sub.add_parser("prepare", help="Write the cleaned, re-split COCO dataset for training")
    p.add_argument("--export", default=str(EXPORT_DIR))
    p.add_argument("--manifest", default=str(MANIFEST))
    p.add_argument("--out", default=str(PREPARED_DIR))
    p.set_defaults(func=_cmd_prepare)

    p = sub.add_parser("summary", help="Print split sizes and class coverage from the manifest")
    p.add_argument("--manifest", default=str(MANIFEST))
    p.set_defaults(func=_cmd_summary)

    return parser


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))
