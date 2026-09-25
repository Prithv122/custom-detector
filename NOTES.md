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

## Evaluation protocol for Run 1 (fixed 2026-09-25, before any confusion count was computed)

Written after seeing the per-class AP table (that is where the hypothesis came from), and
before matching a single prediction to a ground-truth box. Committed on its own so the order
is in the history.

**Question.** On the test date, 0.5 kg loses 0.265 AP and 1.5 kg loses 0.231, while nine other
classes lose 0.04–0.08. Is that because they are labelled as their same-colour heavy partner
(0.5 → 5 kg, white; 1.5 → 15 kg, yellow)?

1. **Matching.** Per image, predictions at or above the confidence threshold are sorted by
   confidence (highest first). Each one takes the still-unmatched ground-truth box with the
   highest IoU, if that IoU is ≥ the threshold. Matching **ignores class**, since a class-aware
   match could never see a confusion. One-to-one: a ground-truth box is matched at most once.
   If a wrong-class box outranks a right-class box on the same plate, the wrong one wins. That
   is the model's top answer.
2. **IoU threshold: 0.5.** mAP@50 is 0.953, so localisation isn't the question; class is.
3. **Outcomes.** Every ground-truth box ends as exactly one of: *correct* (matched, same class),
   *confused as X* (matched, different class), or *missed* (unmatched). Every kept prediction
   is exactly one of: correct, a confusion (counted against its predicted class), or
   *background* (unmatched). Per class: precision = correct / predictions of that class,
   recall = correct / ground truth of that class. So a confusion costs recall for the true
   class and precision for the predicted one.
4. **Confidence threshold.** The single value maximising micro-F1 on **val**, searched over
   0.05–0.95 in steps of 0.01, then used unchanged on test. Test never picks a threshold.
5. **Same-colour confusion.** A ground-truth box of class A matched to a prediction of class B,
   where {A, B} is one of the five pairs: 25/2.5 red, 20/2 blue, 15/1.5 yellow, 10/1 green,
   5/0.5 white. Both directions are reported separately (light → heavy, heavy → light).
6. **Reporting.** One row per true class, on val and on test: correct / same-colour partner /
   other class / missed, as counts and shares of that class's ground truth (each row sums to
   100%), with Wilson 95% intervals on the partner share. About 100 test boxes per light
   class, so the intervals are wide and are shown, not hidden.
7. **Collar (`zacisk`).** Included in matching (a plate called a collar is an other-class
   confusion, and vice versa), reported in its own row, part of no colour pair.
8. **Cross-check.** Class-aware AP@50 is recomputed from the saved predictions. It should land
   near rfdetr's 0.953. A large gap means the predictions file or its coordinates are wrong,
   and nothing below it is trusted.

**Decision rule.** The hypothesis is *supported* only if, for **both** 0.5 kg and 1.5 kg on
test, (a) partner confusion is the largest of their three error types (partner, other class,
missed), (b) its share is higher than the partner share of each control light plate (1 → 10,
2 → 20, 2.5 → 25) on test, and (c) it is higher than the same class's own share on val.
If *missed* is the largest error for both, it's **rejected**: that is a detection (recall)
failure, which points back at scale or appearance, not colour. Anything else is **mixed**
and gets reported as such.

**No second training run** until this is answered. A new run needs a named failure mode to
test, not a hope of a higher score.

## Backdrop check (fixed 2026-09-25, before any pixel was measured)

Written after Run 1's evaluation and after looking at 8 crops per group by eye. Nothing
below has been computed yet. Committed on its own, like the Run 1 protocol, so the order
shows in the history.

**Question.** On the test date, 40 of the 97 ground-truth 1.5 kg plates are read as 0.5 kg,
36 are read correctly and 21 are missed. In the crops, misread plates were mostly in front of
the yellow-green backdrop and correctly read ones were often in front of the dark-blue one.
**Is the backdrop behind a test 1.5 kg plate yellower when the plate is misread as 0.5 kg
than when it is read correctly?**

**Boxes and outcomes.** Every test ground-truth box of class `1.5kg`. Each one's outcome comes
from Run 1's matching exactly (class-agnostic greedy, IoU 0.5) at Run 1's confidence
threshold **0.80**. That threshold was chosen on val and is not picked again. The two groups
compared are **M** = matched to a `0.5kg` prediction and **C** = matched to a `1.5kg`
prediction. Missed and other-class boxes are measured and reported, but they are not part of
the comparison.

**1. Measurement, from pixels and labels only.**
- Pixels: the prepared test image, RGB, 8-bit. Its size must equal the COCO `width`/`height`.
  Any mismatch stops the run: boxes and pixels would not line up.
- Ring: the box grown by a margin `m = max(8, 0.25 × max(w, h))` px on every side, clipped to
  the image. From that area remove **every** ground-truth box in the image (all classes,
  including this one), so neighbouring plates and collars don't count as background. Also
  remove pure-black pixels (0, 0, 0), which is Roboflow's padding on the 640 × 640 images.
  What is left is the ring.
- Colour space: CIELAB, from sRGB with the D65 white point (standard formulas, no library
  colour management).
- Per-box statistic (**primary**): the median **b\*** of the ring pixels. b\* runs from blue
  (negative) to yellow (positive), so it separates the two backdrops seen in the crops, and it
  needs no circular statistics the way hue would.
- Too little background: a box whose ring has fewer than **150** pixels is *unmeasurable*. It
  is counted per group and left out of the statistics.
- **Secondary, reported but not decisive:** the median chroma C\* of the plate itself (the
  central 60 % of the box in each dimension), for the "misread plates look pale" observation.
  It may be a result of the backdrop (auto white balance) rather than a separate cause, so it
  is not part of the verdict.

**2. Summary over boxes.**
- Effect size: **AUC** = P(ring b\* of an M box > ring b\* of a C box), ties counted as ½
  (Mann–Whitney). 0.5 means the backdrop says nothing. Above 0.5 means misread plates sit in
  front of yellower backdrops.
- Uncertainty: boxes in one photo share one backdrop, so they are not independent. **Cluster
  bootstrap over images:** resample the test images that hold at least one measurable M or C
  box, with replacement, **10,000** times, seed **20260925**, and recompute AUC each time. The
  95 % CI is the 2.5th–97.5th percentile. Resamples missing either group are skipped and
  counted.
- Also reported: the medians and IQRs of ring b\* for M, C and missed; box and image counts
  per group; and a **within-image** check over photos that hold both an M and a C box (the
  AUC over M-vs-C pairs from the same photo, and how many photos and pairs that covers). A
  same-photo pair has the same backdrop. If misreads still happen there, the backdrop can't
  be the whole story. This is reported and does not decide the verdict.

**3. Decision rule.**
- **Minimum n:** at least **15** measurable boxes from at least **6** distinct images in each of
  M and C. Otherwise **inconclusive**, whatever the numbers are.
- **For:** AUC ≥ **0.70** and CI lower bound > **0.55**.
- **Reversed:** AUC ≤ 0.30 and CI upper bound < 0.45. The backdrop is associated with the
  misread, but in the opposite direction. Counts as *against* the hypothesis as stated.
- **Against:** the whole CI lies inside [0.30, 0.70], which rules out a strong effect in
  either direction.
- **Inconclusive:** anything else, such as a wide CI straddling 0.5, or a moderate AUC.

**What each answer means.**
- *For* gives Run 2 a named purpose: colour augmentation (hue/saturation jitter) aimed at
  1.5 kg ↔ 0.5 kg, with its own success criterion written down before it trains.
- *Against*, *reversed* or *inconclusive* means no Run 2. The project moves to the demo, HF
  weights and ship. The README reports the 1.5 → 0.5 kg failure as known and unexplained.
- Even *for* is an association on one test date, not a cause. The backdrop may stand in for
  the part of the day, the light, or the physical plates. Only an intervention (Run 2's
  augmentation, or recolouring the backdrop) can test the cause.

**Order:** this protocol → code + synthetic-image tests → one run → numbers committed as they
come out. Nothing above gets edited after the run. Anything learned goes in the log as a
separate, dated entry.

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

### 2026-09-24 (night) — training notebook
- **Built:** `notebooks/train_rfdetr_kaggle.ipynb`. Clones the repo, rebuilds the cleaned split with
  the same `download` + `materialize` code as the CLI, asserts 1,263 / 331 / 402 and that every
  split holds exactly the manifest's files, trains RF-DETR-S, then writes val and test
  predictions as flat CSV rows (file, class name, confidence, box) plus `metrics.csv` and a run
  record (commit, versions, GPU, minutes, config).
- **Decided — data path:** Roboflow key as a Kaggle Secret, dataset rebuilt in the notebook.
  Over an HF mirror: the kept images' photographer isn't verified, so publishing a second copy
  adds exposure for no gain, and the rebuild is already proven exact by the local test. Over a
  private Kaggle dataset: nobody else could reproduce from it.
- **Decided — test discipline:** best checkpoint chosen on `valid` (`best_model_metric="map"`
  default); `run_test=True` scores `test` once at the end. No config change is allowed because
  of a test number.
- **Decided — predictions on disk:** same-colour confusion and per-class P/R are computed from the
  saved predictions, off-GPU, so the evaluation code gets real tests in CI.
- **Not yet known:** time per epoch on a T4, and whether batch 8 × accum 2 fits in 16 GB
  (fallback: 4 × 4). `rfdetr` exposes no seed in `TrainConfig` — one run is one sample; say so
  next to the numbers.
- **Checked:** `src/` compiles on Python 3.11 (Kaggle's image may not be 3.12); the notebook
  imports the modules from `src/` instead of installing the package, which requires 3.12.

### 2026-09-25 — first Kaggle run
- **Ran:** notebook at `6220174`, Tesla T4, torch 2.10.0+cu128, rfdetr 1.11.0, 161.9 min. Split
  rebuilt and membership-checked on Kaggle (1,263 / 331 / 402). 11 classes — placeholder dropped
  as expected. Batch 8 × accum 2 fit in memory.
- **Early stopping:** best EMA val mAP@50–95 0.8293 at epoch 36, but the 0.0008 gain over
  epoch 29 was below `min_delta` 0.001, so the patience counter ran from 29 → stopped after 39.
  The checkpoint is still epoch 36's.
- **Result:** test mAP@50 0.953 / mAP@50–95 0.741; val 0.993 / 0.829. Per class, 9 of 11 drop
  0.04–0.08; **0.5 kg −0.265, 1.5 kg −0.231.**
- **Checked, not the explanation:** ground-truth box height on test is ~0.6–0.7× val for every
  class (portrait, uncropped). 1 kg / 2 kg shrink as much as 0.5 / 1.5 kg but barely drop.
  **Hypothesis to test next:** the two outliers are confused with their same-colour heavy
  partner (5 kg white, 15 kg yellow) — counts from `results/predictions_test.csv`.
- **Broke:** `kaggle kernels output` without a filter died on a 500 MB intermediate `.ckpt`
  (IncompleteRead). `--file-pattern checkpoint_best_total` fetched just the 128 MB model. The
  uv-installed `kaggle` shim is broken on this machine; `uvx --from kaggle kaggle` works.
- **Committed:** `results/` (metrics, config, val/test predictions, run record). Checkpoint kept
  out of git in `models/kaggle-run1/` — goes to the HF Hub for the demo.

### 2026-09-25 — Run 1 evaluated (protocol committed first, `23b6c55`)
- **Order in history:** protocol `23b6c55` → code + fixture tests `aab14e3` → numbers.
- **Cross-check passed:** recomputed mAP@50 val 0.993 / test 0.954 vs rfdetr 0.993 / 0.953.
  Predictions file and coordinates are trustworthy.
- **Threshold 0.80** on val (micro-F1 0.978).
- **Pre-registered verdict: mixed.** Plain reading: **hypothesis wrong.** Partner confusion
  1/100 for 1.5 kg, 0/103 for 0.5 kg, zero everywhere on val. "Mixed" rather than "rejected"
  only because 0.5 kg has no errors of its own — the rule didn't anticipate a class whose AP
  drop is entirely *other classes' boxes landing on it*. Lesson: a decision rule written over
  recall-side errors can't see a precision-side failure. Next rule should cover both.
- **What actually happens:** test 1.5 kg → 0.5 kg ×40, missed ×21, correct ×36. 0.5 kg recall
  1.00, precision 0.715. One error, two AP drops. Size-neighbour, cross-colour confusion.
- Also: val 2 kg → 1 kg ×22 (does not recur on test); 25 kg → 15/20 kg ×6 on test.
- **Exploratory (8 crops per group, model vision):** misread test 1.5 kg plates look pale/olive,
  mostly against the yellow-green backdrop; correctly read ones are saturated yellow, half on
  the dark-blue backdrop. Train 1.5 kg includes pale/cream plates; train 0.5 kg includes beige
  ones — colour already overlaps in training. Crop sheet not committed (unverified
  photographer, same reason as no dataset mirror).
- **Next check (write the rule first):** does the backdrop behind a 1.5 kg plate predict whether
  it is misread? Needs a per-box background measure (e.g. mean hue of a ring around the box)
  and a threshold fixed before counting.
- **Bug fixed on the way:** `evaluate` crashed printing `→` on a Windows cp1252 console after
  the files were written. Report text is ASCII now.

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
| HF / Kaggle mirror of the cleaned images | Re-publishes photos whose author isn't verified; the manifest + pinned export already rebuild the set exactly. |

## Open questions

- [x] RF-DETR and the placeholder category `objects` (id 0): `rfdetr` 1.11.0 drops an unannotated
      category that other categories name as their supercategory (`filter_parent_categories` in
      `rfdetr/datasets/coco.py`), and maps val/test labels through the train split's mapping.
      Leave the exported categories as they are; the notebook pins 1.11.0.
- [x] Kaggle phone verification — done 2026-09-25; first run completed.
- [x] How the data reaches Kaggle: rebuilt inside the notebook from the pinned export + the
      committed manifest, with the Roboflow key as a Kaggle Secret. No HF mirror (see log).
- [ ] Ask the dataset owner who took the phone photos? Only matters if the demo goes beyond a
      portfolio.
