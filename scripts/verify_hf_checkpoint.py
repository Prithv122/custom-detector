"""Check that the published 2A checkpoint is the accepted one and that it still works.

Downloads the checkpoint from the Hugging Face Hub at a pinned revision into an empty cache,
compares its SHA-256 with the local file's, then runs CPU inference with the downloaded copy on
the 100 test-date 1.5 kg boxes and compares each box's outcome with the Kaggle 2A predictions in
``results/run2a/`` (same matching and threshold as ``run2-eval``). Writes the evidence to
``results/hf_publication.json``.

    uv run python scripts/verify_hf_checkpoint.py --revision <hub commit>

Needs the prepared dataset (``custom-detector prepare``) and network access.
"""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from importlib.metadata import version
from pathlib import Path

from huggingface_hub import hf_hub_download

from detector.evaluate import load_ground_truth, load_predictions
from detector.recolour_run import load_model, load_rgb, predict, sha256_file
from detector.run2_eval import count, target_outcomes

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "Prithv122/custom-detector"
FILENAME = "checkpoint_best_total.pth"
EXPECTED_SHA256 = "f802956130702712f05bfb82d8b437ea1648efe642ff90bbba2fb38a1d8bce1f"
LOCAL_CHECKPOINT = ROOT / "models" / "kaggle-run2a" / "run" / FILENAME


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--revision", required=True, help="Hub commit to verify (full SHA)")
    ap.add_argument("--prepared", default=str(ROOT / "data" / "processed" / "plates_v10_clean"))
    ap.add_argument("--out", default=str(ROOT / "results" / "hf_publication.json"))
    ap.add_argument("--tmp", default=None, help="parent for the throwaway download cache")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(dir=args.tmp) as cache:
        path = Path(hf_hub_download(REPO_ID, FILENAME, revision=args.revision, cache_dir=cache))
        size = path.stat().st_size
        downloaded_sha = sha256_file(path)
        if downloaded_sha != EXPECTED_SHA256:
            raise SystemExit(f"SHA-256 mismatch: downloaded {downloaded_sha}")
        model = load_model(path, device="cpu")

    test_dir = Path(args.prepared) / "test"
    coco = json.loads((test_dir / "_annotations.coco.json").read_text())
    gt = load_ground_truth(test_dir / "_annotations.coco.json")
    gt = {name: boxes for name, boxes in gt.items() if any(c == "1.5kg" for c, _ in boxes)}
    threshold = json.loads((ROOT / "results" / "run2_evaluation.json").read_text())["runs"]["2A"][
        "threshold"
    ]

    local_preds = {name: predict(model, load_rgb(test_dir, coco, name)) for name in gt}
    kaggle_preds = load_predictions(ROOT / "results" / "run2a" / "predictions_test.csv")
    local = target_outcomes(gt, local_preds, threshold)
    kaggle = target_outcomes(gt, kaggle_preds, threshold)
    mismatches = {
        i: {"kaggle": kaggle[i], "local": local[i]} for i in local if local[i] != kaggle[i]
    }

    report = {
        "repo_id": REPO_ID,
        "revision": args.revision,
        "file": FILENAME,
        "size_bytes": size,
        "sha256_expected": EXPECTED_SHA256,
        "sha256_downloaded": downloaded_sha,
        "sha256_local": sha256_file(LOCAL_CHECKPOINT) if LOCAL_CHECKPOINT.exists() else None,
        "sha256_match": downloaded_sha == EXPECTED_SHA256,
        "inference": {
            "device": "cpu",
            "threshold": threshold,
            "boxes": len(local),
            "counts_downloaded_cpu": count(local),
            "counts_kaggle_gpu": count(kaggle),
            "outcome_mismatches": mismatches,
        },
        "versions": {
            "python": platform.python_version(),
            "rfdetr": version("rfdetr"),
            "torch": version("torch"),
            "huggingface_hub": version("huggingface_hub"),
        },
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
