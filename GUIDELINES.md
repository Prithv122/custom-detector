# Custom Object Detector — H2

**Tier:** 2 · **Category:** Computer Vision · **Wave:** Wave 3 — Production tier

Root rules in `../GUIDELINES.md` apply. This file is project-specific only — keep it under 40 lines.

## What this is

Weightlifting-plate detector (10 weights + collar) on a **public dataset**: ProjektCiezary
*Weightlifting Plates*, Roboflow v10, CC BY 4.0. Audited and cleaned (1,996 of 2,251 kept) and
re-split without leakage (test = whole date 2025-05-04). RF-DETR-S fine-tuned on Kaggle.
Detection metrics only — the loaded-weight metric was dropped (no independent ground truth).
Full design, audit findings, superseded own-photo plan: `NOTES.md`.

## Stack

Python 3.12 · `rfdetr` (Apache-2.0, sizes N/S/M/L only — XL/2XL are PML 1.0) · Roboflow as
dataset source · Kaggle GPU for training → weights on HF Hub → local CPU inference.
**No Ultralytics** (AGPL-3.0).

## Acceptance criteria

- [x] Dataset cleaned + leakage-free split, committed as `data/manifest.csv`, invariants tested
- [x] RF-DETR-S fine-tuned; mAP@50 / mAP@50–95, per-class AP/P/R on the held-out date
- [x] Same-colour-pair confusion analysis; val → test gap reported
- [x] Demo app: `custom-detector demo` (local, Gradio, 2A from the Hub); no hosted URL
- [x] Ship gate passes

## Project-specific notes

- `data/manifest.csv` is the source of truth for exclusions and split — never hand-edit;
  rebuild with `custom-detector manifest` (the local test proves it reproduces exactly).
- Export in `data/raw/projektciezary_v10_coco/`, prepared set in `data/processed/` — both
  git-ignored. Key in `.env` (`ROBOFLOW_API_KEY`), never printed.
- Kaggle: verified. CLI via `uvx --from kaggle kaggle` (uv shim broken); use `--file-pattern`.
- Recolouring test verdict: **for (scene)** (2026-09-30, NOTES.md). Whole-scene recolour flips
  60% of misreads; ring-only recolour flips only 5% (mostly turns into missed detections, not
  a class change). Run 2 scored **fixed** (`26864ee`, write-up `880f9bb`): ship 2A. 2A is on
  HF `Prithv122/custom-detector` @ `be16623` (`50abd89`, round trip verified). The demo is local
  only (`demo` extra, default threshold 0.53). The test date is burned; Run 1's 0.741 is the
  new-day estimate.
- **No example photos are shipped.** The kept images are third-party photos with an unverified
  author, so the demo reads a local folder through `--examples`.
