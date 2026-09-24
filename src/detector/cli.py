"""Console entry point for the Gym A data pipeline: sort photos, validate loads, upload."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from detector.loads import validate_loads_csv
from detector.roboflow_upload import build_roboflow_project, upload_gym_a
from detector.sort_photos import default_ask_load_id, sort_by_slate

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _cmd_sort_photos(args: argparse.Namespace) -> int:
    moved = sort_by_slate(
        Path(args.source),
        Path(args.dest),
        gap_seconds=args.gap_seconds,
        ask_load_id=default_ask_load_id,
    )
    print(f"Sorted {len(moved)} photo(s) into {args.dest}")
    return 0


def _cmd_validate_loads(args: argparse.Namespace) -> int:
    report = validate_loads_csv(Path(args.csv), Path(args.protocol))
    for issue in report.issues:
        where = f"row {issue.row}" if issue.row is not None else "file"
        print(f"[{issue.level.upper()}] {where}: {issue.message}")
    print(f"\n{len(report.errors)} error(s), {len(report.warnings)} warning(s)")
    return 0 if report.ok else 1


def _cmd_upload(args: argparse.Namespace) -> int:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("ROBOFLOW_API_KEY is not set (see .env.example)", file=sys.stderr)
        return 1

    project = build_roboflow_project(api_key, args.workspace, args.project)
    result = upload_gym_a(Path(args.source), Path(args.protocol), project)

    for split, count in sorted(result.uploaded.items()):
        print(f"{split}: {count} uploaded")
    for path, error in result.failures:
        print(f"FAILED {path}: {error}", file=sys.stderr)
    return 1 if result.failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="custom-detector")
    sub = parser.add_subparsers(dest="command", required=True)

    sort_p = sub.add_parser(
        "sort-photos", help="Sort a phone-dump folder into data/raw/gym_a by slate"
    )
    sort_p.add_argument("source", help="Folder the phone photos were copied into")
    sort_p.add_argument("--dest", default=str(DATA_DIR / "raw" / "gym_a"))
    sort_p.add_argument("--gap-seconds", type=float, default=120.0)
    sort_p.set_defaults(func=_cmd_sort_photos)

    validate_p = sub.add_parser(
        "validate-loads", help="Check data/loads.csv against the capture protocol"
    )
    validate_p.add_argument("--csv", default=str(DATA_DIR / "loads.csv"))
    validate_p.add_argument("--protocol", default=str(DATA_DIR / "CAPTURE_PROTOCOL.md"))
    validate_p.set_defaults(func=_cmd_validate_loads)

    upload_p = sub.add_parser(
        "upload", help="Upload data/raw/gym_a to Roboflow with the pre-fixed split"
    )
    upload_p.add_argument("--source", default=str(DATA_DIR / "raw" / "gym_a"))
    upload_p.add_argument("--protocol", default=str(DATA_DIR / "CAPTURE_PROTOCOL.md"))
    upload_p.add_argument("--workspace", required=True)
    upload_p.add_argument("--project", required=True)
    upload_p.set_defaults(func=_cmd_upload)

    return parser


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))
