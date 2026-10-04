"""Console entry point: download the export, build the manifest, prepare the data, evaluate."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from detector.backdrop import format_summary, measure_split, measures_to_rows, summarise
from detector.dataset import CLASSES
from detector.dataset.manifest import build_manifest, read_manifest, write_manifest
from detector.dataset.prepare import download, materialize
from detector.dataset.split import assign_splits, cross_split_near_duplicates
from detector.demo import (
    DEFAULT_THRESHOLD,
    MAX_THRESHOLD,
    MIN_THRESHOLD,
    build_app,
    example_images,
)
from detector.demo import load as load_demo_model
from detector.evaluate import evaluate, format_report
from detector.recolour_run import arm_rows
from detector.recolour_run import format_summary as format_recolour_summary
from detector.recolour_run import run as run_recolour
from detector.run2_eval import evaluate_run2
from detector.run2_eval import format_report as format_run2_report

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
EXPORT_DIR = DATA_DIR / "raw" / "projektciezary_v10_coco"
MANIFEST = DATA_DIR / "manifest.csv"
PREPARED_DIR = DATA_DIR / "processed" / "plates_v10_clean"
RESULTS_DIR = DATA_DIR.parent / "results"
CHECKPOINT = DATA_DIR.parent / "models" / "kaggle-run1" / "run" / "checkpoint_best_total.pth"


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


def _cmd_evaluate(args: argparse.Namespace) -> int:
    report = evaluate(Path(args.prepared), Path(args.results))
    out = Path(args.results)
    (out / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    text = format_report(report)
    (out / "evaluation.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _cmd_run2_eval(args: argparse.Namespace) -> int:
    report = evaluate_run2(Path(args.prepared), Path(args.results))
    out = Path(args.results)
    (out / "run2_evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    text = format_run2_report(report)
    (out / "run2_evaluation.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _cmd_backdrop(args: argparse.Namespace) -> int:
    measures = measure_split(
        Path(args.prepared) / "test", Path(args.results) / "predictions_test.csv"
    )
    summary = summarise(measures)
    out = Path(args.results)
    rows = measures_to_rows(measures)
    with (out / "backdrop_boxes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "backdrop.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = format_summary(summary)
    (out / "backdrop.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _cmd_recolour(args: argparse.Namespace) -> int:
    report, arm_outcomes = run_recolour(
        Path(args.prepared), Path(args.results), Path(args.checkpoint)
    )
    out = Path(args.results)
    (out / "recolour.json").write_text(json.dumps(report, indent=2) + "\n")
    text = format_recolour_summary(report)
    (out / "recolour.md").write_text(text + "\n", encoding="utf-8")
    if arm_outcomes:
        rows = arm_rows(arm_outcomes)
        with (out / "recolour_boxes.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(text)
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    if not MIN_THRESHOLD <= args.threshold <= MAX_THRESHOLD:
        print(f"--threshold must be between {MIN_THRESHOLD} and {MAX_THRESHOLD}", file=sys.stderr)
        return 1
    if importlib.util.find_spec("gradio") is None:
        print("The demo needs Gradio: run `uv sync --extra demo`", file=sys.stderr)
        return 1
    examples = example_images(Path(args.examples)) if args.examples else None
    model = load_demo_model(Path(args.checkpoint) if args.checkpoint else None)
    app = build_app(model, args.threshold, examples)
    app.launch(server_name="127.0.0.1", server_port=args.port, share=False)
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

    p = sub.add_parser("evaluate", help="Score saved predictions (needs the prepared dataset)")
    p.add_argument("--prepared", default=str(PREPARED_DIR))
    p.add_argument("--results", default=str(RESULTS_DIR))
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser("run2-eval", help="Score Run 2: 2C vs 2A, per the NOTES.md protocol")
    p.add_argument("--prepared", default=str(PREPARED_DIR))
    p.add_argument("--results", default=str(RESULTS_DIR))
    p.set_defaults(func=_cmd_run2_eval)

    p = sub.add_parser("backdrop", help="Backdrop check on test 1.5 kg plates (NOTES.md)")
    p.add_argument("--prepared", default=str(PREPARED_DIR))
    p.add_argument("--results", default=str(RESULTS_DIR))
    p.set_defaults(func=_cmd_backdrop)

    p = sub.add_parser("recolour", help="Baseline gate + the five-arm recolouring test (NOTES.md)")
    p.add_argument("--prepared", default=str(PREPARED_DIR))
    p.add_argument("--results", default=str(RESULTS_DIR))
    p.add_argument("--checkpoint", default=str(CHECKPOINT))
    p.set_defaults(func=_cmd_recolour)

    p = sub.add_parser(
        "demo", help="Local web demo on the published 2A checkpoint (needs --extra demo)"
    )
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--checkpoint", help="local copy of the 2A checkpoint (hash-checked)")
    p.add_argument("--examples", help="folder of photos to offer as clickable examples")
    p.set_defaults(func=_cmd_demo)

    p = sub.add_parser("summary", help="Print split sizes and class coverage from the manifest")
    p.add_argument("--manifest", default=str(MANIFEST))
    p.set_defaults(func=_cmd_summary)

    return parser


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))
