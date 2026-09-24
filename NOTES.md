# Build Notes — Custom Object Detector

Working notes: what broke, what you tried, why you chose X over Y.
Not for recruiters — for you, six months from now, in an interview.

Keep it rough. Rough is the point.

---

## Settled design (revised 2026-09-24)

Replaces the own-photo design of 2026-09-23 (kept below under *Superseded design*).

| Decision | Chose | Why |
|---|---|---|
| Domain | Weight plates, side-on views of a loaded sleeve | Not a COCO class. Same domain as the original plan, but the images are now colour-coded competition/bumper plates on a loading rig, not iron gym plates at 30–45°. The README says so plainly. |
| Dataset | ProjektCiezary *Weightlifting Plates*, Roboflow Universe, **v10 pinned** | Per-weight boxes already exist (2,251 images). Own capture (~225 photos + labelling) was the slowest step of the old plan. |
| Cleaning | Exclude 255 of 2,251 by rule → **1,996 kept** | 225 `ZE_*` competition-video crops + 25 press photos (third-party imagery the uploader's CC BY can't license), 3 images with visible but unboxed plates, 2 with one plate labelled as two weights. |
| Classes | All 10 plate weights (`0.5kg`…`25kg`) + `zacisk` (collar), as labelled | Dropping 0.5–2 kg would leave visible plates unboxed in ~1,960 images, i.e. teach the model they are background. Every class has ≥149 train and ≥70 test images. |
| Split | **Test = whole capture date 2025-05-04; val = whole sessions; train = rest** → 1,263 / 331 / 402 | The published split leaks (below). Test is a day the model never saw. Rule-based, from dataset support only — see `detector.dataset.split`. |
| Source of truth | `data/manifest.csv` (committed, one row per export image) | Holds hashes, boxes per class, exclusion reason, session and split. CI checks the split invariants from it without the images; a local test proves it rebuilds byte-for-byte from the export. |
| Detector | RF-DETR-S (`rfdetr`, Apache-2.0) | Unchanged. Repo stays MIT. |
| Metrics | Detection only (below) | The loaded-weight metric is gone: it needed a load recorded independently of the labels. Summing labelled plates and scoring against that sum would re-score the detector against its own labels. |

**Metrics** (planned; nothing trained yet):
- mAP@50 and mAP@50–95 on the held-out date — the headline pair.
- Per-class AP, precision, recall — over the 10 plates; the collar reported on its own.
- Confusion between same-colour pairs (25/2.5 red, 20/2 blue, 15/1.5 yellow, 10/1 green,
  5/0.5 white): the model must separate these mostly by size.
- Val → test gap: val shares dates with train, test doesn't. A large gap = the model learned the
  day's background, not the plates.
- Not supported by the data: small-object analysis (3% of boxes are COCO-small) and an occlusion
  breakdown (no occlusion flag in the labels).

## Dataset audit (2026-09-24)

Programmatic over all 2,251 images; model vision only on 28 flagged or sampled images.

- **Leakage in the published split.** 256-bit pHash: 1,117 images have a near-identical twin
  (distance ≤ 20); for 315 of them the twin sits in a different split. Cause: each session is a
  series of shots of one rig with one small plate swapped between shots — `…120057` (train) and
  `…120030` (valid) are the same bar, 27 s apart. No exact duplicates (SHA-256).
- **Near-duplicates never cross dates.** A 64-bit pHash flagged 1,311 cross-date pairs; all were
  false positives (256-bit: closest cross-date pair is 32). So date/session grouping fixes it.
- **Provenance.** 2,001 phone-timestamped images across 7 dates (May–Nov 2025), same gym
  backgrounds → consistent with the uploader's own photos (inference, not proof). `ZE_*`: cropped,
  upscaled competition video — sponsor boards, athletes' arms. Numbered files (`31.webp`…): press
  photos from Olympic / Beijing 2008 / Santiago 2023 events, one with an agency credit.
- **Classes are tied to dates.** e.g. no 15 kg on 2025-05-01, 09-29, 10-09; no 5 kg on 3 dates.
  A split by date must be checked for class coverage — the test-date rule does exactly that.
- **Domain shift in test.** 2025-05-04 is the only date shot portrait (480×640) and uncropped,
  with a distinctive backdrop; other dates are cropped square. Report it next to the numbers.
- Clean split check: no cross-split pair within 30 bits; closest are 32 (train/val), 84
  (train/test), 82 (val/test).

## Licence check

- Dataset: README and every image's COCO licence field say **CC BY 4.0** (v10, owner
  "ProjektCiezary", no description or author named). Attribution + "changes made" notice go in
  the README. We don't claim to have verified who shot the kept images.
- Ultralytics YOLO: **AGPL-3.0**, weights included; README sells an Enterprise Licence for
  production and internal use. Excluded — a public demo would pull the repo onto the AGPL path.
- `rfdetr` 1.10.1: Apache-2.0 (GitHub + PyPI). DINOv2 backbone: Apache-2.0.
  **RF-DETR-XL/2XL are PML 1.0**, not Apache — excluded.
- Exporting to ONNX changes the runtime, not the licence of the model inside it. Not treated
  as a way around anything.
- Open caveat, not resolved: pretrained weights come from COCO. COCO annotations are
  CC-BY-4.0, but the Flickr images carry mixed licences, and whether trained weights inherit
  those terms is unsettled. Document it under References.

---

## Log

### 2026-09-23
- **Tried:** 7 candidate domains (plates, document marks, currency, FMCG shelf SKUs, PCB
  components, Indian road hazards, candlestick patterns).
- **Broke:** nothing yet — design only.
- **Learned:** in the load schedule several different plate sets add up to the same total
  (70 kg three ways, 90 kg three ways). An exact-total match can hide wrong plates.

### 2026-09-24 (morning) — capture pipeline
- **Built ahead of the Gym A visit** (photos still on phone, `loads.csv` still empty):
  a protocol parser reading the load schedule straight out of `CAPTURE_PROTOCOL.md`, a
  `loads.csv` validator, a burst sorter grouping a phone dump by capture-time gap, and an
  uploader that resolved each load's split from the schedule. All retired the same day
  (commit `ac1b874` has them).
- **Tried:** auto-detecting the slate card with OCR. Rejected — a misread slate would silently
  corrupt the answer key.

### 2026-09-24 (evening) — switched to a public dataset
- **Tried:** searched Roboflow Universe, Kaggle and Hugging Face for per-weight plate boxes.
  Kaggle/HF have none. One Roboflow set qualified (ProjektCiezary); the rest were one generic
  "plate" class, colour labels, or under ~500 images. Alternatives shortlisted in case it failed
  the audit: Aquarium (CC BY 4.0, 638 images, penguin/puffin overlap COCO "bird") and Laboro
  Tomato (CC BY-NC-SA 4.0).
- **Broke:** the published split — see *Dataset audit*. Its own hosted model (mAP@50 52%) was
  trained and scored on that leaky split, so it is not a baseline we can quote.
- **Learned:** a 64-bit perceptual hash is too coarse for a fixed-rig scene — every close-up of a
  bar sleeve looks alike at 8×8. 256 bits separated same-session twins (≤ 20) from different-day
  shots (≥ 32) cleanly.
- **Learned:** the burst-by-time-gap idea from the retired sorter carried straight over — it is
  now how sessions are formed for the split.

---

## Superseded design (2026-09-23) — kept for the record

Own photographs at two gyms, 6 classes (`25kg`…`2.5kg`), a split fixed by load before capture,
Gym B as a test-only domain-shift set, and a second metric layer — exact loaded weight, exact
plate set per sleeve, mean kg error — scored against the load **recorded at capture** in
`loads.csv`, never derived from labels. Capture protocol and answer-key template:
`data/archive/gym-capture/`. Dropped because collection was the bottleneck; the loaded-weight
layer could not survive the switch (no independent ground truth in a public dataset).

## Rejected approaches

| Approach | Why rejected |
|---|---|
| Own gym photography (Gym A/Gym B) | Superseded 2026-09-24 — ~225 photos + labelling for a result a public dataset now gives. |
| ProjektCiezary's published split | Leaks: 315 images have a near-identical twin in another split. |
| Keeping `ZE_*` and press images | Third-party footage/photos; the uploader's CC BY can't license them. |
| Six classes (dropping 0.5–2 kg) | Would leave visible plates unboxed = trained as background. Merge, don't drop, if ever reduced. |
| Loaded-weight metric on the public set | No independent ground truth; it would just re-score the labels. |
| Document marks (signature/stamp/QR) | Real documents carry personal data and can't be published. |
| Candlestick chart patterns | Boxes are subjective → mAP measures labeller consistency, not the model. |
| Indian currency notes | Common Kaggle project; mAP likely saturates near 0.95 and proves little. |
| FMCG shelf SKUs, PCB parts, road hazards | Fine-grained classes, tiny objects or heavy occlusion. |
| Ultralytics YOLO | AGPL-3.0 (see licence check). |

## Open questions

- [ ] Does RF-DETR's COCO loader accept the Roboflow placeholder category `objects` (id 0) as-is,
      or does it need dropping? Check before the first training run.
- [ ] Kaggle phone verification (GPU quota) — confirm before training.
- [ ] Mirror the cleaned 1,996-image subset to the Hugging Face Hub (CC BY permits it, with
      attribution) so a clean clone needs no Roboflow key? Decide before shipping.
- [ ] Ask the dataset owner who took the phone photos? Only matters if the demo goes beyond a
      portfolio.
