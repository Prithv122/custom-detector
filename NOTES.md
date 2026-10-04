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

## Recolouring test (fixed 2026-09-25, before any image was altered)

Written after the backdrop check came out *for*. No image has been recoloured and no inference
has been run locally. The only numbers looked up while writing this are group sizes from
`results/backdrop_boxes.csv`, needed to size the arms.

**Question.** The backdrop check found an association: all 40 test 1.5 kg plates read as
0.5 kg sit in front of a yellow backdrop. **If only the backdrop colour changes, with the same
checkpoint, the same image and the same plate pixels, does the read change?** Yellow → blue
should turn misreads into correct reads. Blue → yellow should do the opposite.

**Units.** Every test photo holds exactly one 1.5 kg plate, so one box is one image. Boxes are
independent and plain binomial counts are valid. Groups come from Run 1's outcomes
(`backdrop_boxes.csv`), and "blue side" / "yellow side" means ring median b\* ≤ 0 or > 0.
The sign of b\* is the neutral point of the colour space, not a cut-off tuned on the data.
- **M** = misread as 0.5 kg: 40 boxes, all on the yellow side.
- **C-blue** = read correctly, blue side: 19 boxes.
- **C-yellow** = read correctly, yellow side: 17 boxes.
- Missed (21) and other-class (3) boxes are not in any arm.

**1. The transformation.**
- Input: the prepared test image, RGB 8-bit, converted to CIELAB exactly as in the backdrop
  check (sRGB, D65, `srgb_to_lab`). The ring is `ring_mask` from the backdrop check, unchanged:
  margin `max(8, 0.25 × max(w, h))`, every ground-truth box removed, pure black removed.
- Recolour: for every pixel in the mask, **keep L\*** and set (a\*, b\*) to the target colour.
  Convert back to sRGB, clip to [0, 255] and round to 8-bit. Keeping L\* keeps the texture,
  edges and shading, so the only thing changed is colour. There's no per-pixel "is it yellow"
  rule: every ring pixel is recoloured.
- **Target colours, fixed by rule and not by outcome.** For each test 1.5 kg box, take the
  median a\* and b\* of its ring. *Blue target* = the median of those over boxes with ring
  b\* ≤ 0. *Yellow target* = the same over boxes with ring b\* > 0. Both use every 1.5 kg box
  on that side, whatever its outcome. Report the two targets and the share of pixels clipped.
- Pixels outside the mask are copied unchanged. Plate pixels are never touched.
- The altered image goes to the model **in memory**. It is never saved as a JPEG, because
  re-compression would change pixels outside the mask.

**2. Arms.**

| Arm | Boxes | Mask | Target | Role |
|---|---|---|---|---|
| **Ring** | M (40) | ring | blue | **primary** |
| **Sham** | M (40) | ring | the pixel's own (a\*, b\*), a Lab round trip | noise floor |
| **Whole** | M (40) | every pixel outside all ground-truth boxes (pure black removed) | blue | reads a null ring result |
| **Reverse** | C-blue (19) | ring | yellow | the opposite direction |
| **Keep** | C-yellow (17) | ring | blue | does recolouring break correct reads? |

**3. Inference and outcome.**
- Model: the Run 1 checkpoint (`models/kaggle-run1/run/checkpoint_best_total.pth`),
  RF-DETR-S at resolution 512. Run locally on CPU with **rfdetr 1.11.0**, the same version as
  the Kaggle run. Versions, device and checkpoint SHA-256 go in the output.
- Each box's outcome uses Run 1's matching exactly: class-agnostic greedy, IoU 0.5,
  confidence threshold **0.80**, over all predictions in the image. There are four possible
  outcomes: `1.5kg`, `0.5kg`, `missed`, `other`.
- **Flip** = the outcome becomes `1.5kg` (Ring, Sham, Whole), or `0.5kg` (Reverse). Any other
  change is counted and reported separately and is never a flip. For Keep, count how many stay
  `1.5kg`.
- **Baseline gate, run first.** Run the same local CPU pipeline on the *unaltered* images. The
  outcomes must match Run 1's for all 76 boxes in the arms (M + C-blue + C-yellow). A box whose
  baseline differs from Run 1 is dropped from every arm and reported. **More than 2 such boxes
  → stop, no verdict.** The CPU run doesn't reproduce the GPU run closely enough to compare.
  Every flip is measured against the *local* baseline, never against the Kaggle predictions.
- Secondary, not decisive: for each M box, the highest confidence of any `1.5kg` prediction
  with IoU ≥ 0.5 to the box, at any confidence, baseline vs Ring. Report the median change.

**4. Decision rule** (counts out of the M boxes that pass the gate, called n below: 40 if none
are dropped).
- **Pipeline noise:** Sham flips > 2 → no verdict. Find the pipeline bug first.
- **For (local):** Ring flips ≥ n/2 (20 of 40), and Sham flips ≤ 2.
- **For (scene):** Ring flips < n/2, Whole flips ≥ n/2, Sham flips ≤ 2. The backdrop matters,
  but only as the colour of the whole scene, not just the area next to the plate.
- **Against:** Ring and Whole flips both ≤ n/10 (4 of 40), and Sham flips ≤ 2.
- **Inconclusive:** anything else.
- Reported with Wilson 95 % CIs next to the verdict, but not deciding it: Reverse flips out of
  19 and Keep out of 17. If *for* comes with Reverse flips near zero, the effect goes one way
  only. If Keep loses reads, recolouring damages images in general and a *for* is weaker. Say
  both plainly.

**What each answer means.**
- *For* (either kind): the backdrop colour causes the misread with this checkpoint on this
  date. Run 2 gets its target, colour augmentation, with the success criterion already listed
  in the log (test 1.5 kg recall on yellow-backdrop plates, 0.5 kg precision, no other class
  outside its Run 1 CI). That criterion is re-committed on its own before training.
- *Against*: the association doesn't carry over to an intervention on surrounding colour. No
  Run 2 on this basis. The README reports the failure, the association, and the negative
  intervention. Move on to the demo and ship.
- *Inconclusive* or *pipeline noise*: no Run 2. Report as it stands and ship.

**What this can't show.**
- Plate pixels stay untouched, so the test covers only "yellow surroundings confuse the
  model". It can't test the other route: yellow light shifting the camera's white balance and
  making the plate itself look pale. *Against* does not rule that out.
- Keeping L\* makes the blue lighter than the real dark-blue backdrop. The recoloured
  images are a partial move toward blue, not a copy of it.
- One checkpoint and one test date. A causal answer here is about this model, not
  about detectors in general.

**Order:** this protocol → code + synthetic-image tests (the mask is the only place pixels
change, L\* is kept within rounding, the Sham round trip moves no channel by more than 1, the
targets come out of the rule) → baseline gate → one run of all five arms → output committed
unedited. Nothing above gets edited after the run. Anything learned goes in the log as a
separate, dated entry.

## Run 2 — colour-cast augmentation (fixed 2026-10-01, before any Run 2 training)

Written after the recolouring test came out *for (scene)*. Nothing has trained. While writing
this I checked two things: how `rfdetr` 1.11.0 routes augmentation configs (its source), and
what the chosen transform does to a synthetic neutral-grey image (numbers below). No test
image was looked at.

This replaces the draft criterion in the recolouring protocol (yellow-side 1.5 kg recall,
0.5 kg precision, no class outside its Run 1 CI). The primary metric is now the misread count
over all 100 test 1.5 kg boxes, because that count *is* the failure. Yellow-side counts and
0.5 kg precision move to secondary. "Outside its Run 1 CI" needed a run-to-run variance that
nobody has measured, so it is replaced by fixed margins against a control run.

**Question.** Does training with a global colour-temperature augmentation reduce the test-date
1.5 kg → 0.5 kg misread, without costing overall detection quality?

**1. The augmentation, and why this one.**
- The recolouring test changed the chroma (a\*, b\*) of the whole scene and kept L\*. So the
  augmentation changes chroma, not lightness.
- Plate colour is part of the label. 1.5 kg is yellow, 0.5 kg white, 10 kg green, and 1.5 and
  0.5 kg are close in size, so colour is what separates them. A hue rotation would push a
  yellow plate toward green while its label stays `1.5kg`, which teaches the model that colour
  means nothing. **Hue/saturation jitter is rejected** for that reason.
- A colour-temperature cast moves every pixel the same way, plate and backdrop together, the
  way white balance or the light does. The plate's colour *relative to the scene* survives.
  That matches the diagnosis: the colour of the whole scene, not a patch next to the plate.
- **Chosen:** Albumentations `PlanckianJitter`, `mode="cied"` (CIE D illuminant series),
  `temperature_limit=(4000, 15000)`, `sampling_method="uniform"`, `p=0.5`. These are all the
  library's defaults for this mode. Nothing is tuned.
- **Measured on neutral grey** (sRGB 128, L\* 53.6): 4000 K → b\* +39.1, L\* 56.2.
  6500 K → b\* +0.4. 15000 K → b\* −34.8, L\* 53.0. That is a b\* span of 74, wider than the
  ~50 between the test date's yellow ring (median b\* 45.2) and its correctly read plates
  (−4.7). Lightness moves at most 2.6 L\*. On a synthetic random image, at most 8.2 % of
  channel values clip.
- **`cied` over `blackbody`:** blackbody at 3000 K raises L\* by 7.8 on the same grey and clips
  46 % of the random image's values. That is a brightness change as much as a colour change.
- **Not added:** brightness, contrast, value, saturation, hue, gamma, blur. One colour
  transform, so a result can be pinned on it.

**2. The confound, and why there are two runs.**
- Run 1 used `aug_config=None`. That selects `rfdetr`'s torchvision pipeline: horizontal flip
  (p 0.5) and resizing with `BILINEAR` + antialias. **Any non-empty `aug_config` switches to
  the Albumentations pipeline.** The flip is the same, but resizing is cv2 `INTER_LINEAR`
  without antialias, and `rfdetr`'s own warning says mAP may drift. Run 1 also had no seed.
  So Run 1 vs a colour run would change three things at once: colour, resize backend, seed.
- So there are two runs on one commit, identical except for the colour transform:

| Run | `aug_config` | Role |
|---|---|---|
| **2C** control | `{"HorizontalFlip": {"p": 0.5}}` | Run 1's augmentation, on the Albumentations path |
| **2A** colour | `{"HorizontalFlip": {"p": 0.5}, "PlanckianJitter": {"mode": "cied", "temperature_limit": (4000, 15000), "sampling_method": "uniform", "p": 0.5}}` | treatment |

- **The primary comparison is 2A vs 2C.** Run 1 is reported next to both as a reference.
- 2C also answers a question of its own: does the failure survive a retrain at all? Run 1 is
  one sample.
- Cost: about 2 × 162 T4 minutes.

**3. Frozen for both runs.** Identical to Run 1 unless listed here.
- Data: `data/manifest.csv` as committed. The split is 1,263 / 331 / 402, rebuilt and
  membership-checked in the notebook exactly as for Run 1.
- Model: RF-DETR-S from the COCO pretrained weights, resolution 512. `rfdetr` 1.11.0 and
  `albumentations` 2.0.8, both pinned.
- The notebook's `TRAIN` dict is unchanged: 50 epochs, batch 8 × accumulation 2, lr 1e-4,
  early stopping (patience 10), `run_test=True`, per-class metrics. Every other `TrainConfig`
  field stays at the value recorded in `results/training_config.json` (encoder lr 1.5e-4,
  EMA, per-batch multi-scale, `scale_jitter`, `min_delta` 0.001, `best_model_metric="map"`).
- Added in both runs: `augmentation_backend="albumentations"`, pinned so the `"cpu"`
  auto-pick can't pick a different backend on Kaggle's image, and `seed=20261001`.
- Hardware: Kaggle, one Tesla T4 (`CUDA_VISIBLE_DEVICES=0`), as for Run 1.
- Outputs: `results/run2c/` and `results/run2a/`, holding the same files as Run 1
  (val/test predictions, `metrics.csv`, `training_config.json`, `run_record.json`).
- **Config check:** each saved `training_config.json` is compared field by field with Run 1's.
  Any difference outside `aug_config`, `augmentation_backend`, `seed` and the paths **voids
  that run**, whatever its numbers.

**4. Checkpoint, threshold, test.**
- Checkpoint: `checkpoint_best_total.pth`, which `rfdetr` picks on val mAP@50–95, as in Run 1.
  Test plays no part in it.
- Confidence threshold: each run's own, by Run 1's rule: the value that maximises micro-F1 on
  that run's val, over 0.05–0.95 in steps of 0.01.
- Test is scored once per run. After any test number has been seen, there is no retraining, no
  config change and no second seed. The one allowed rerun is an infrastructure failure (a
  crash, out of memory, a Kaggle timeout) before the run finishes, logged with its reason.
- Both runs finish before either one's test predictions go through the evaluation.

**5. Primary metric.**
- Unit: the **100** test ground-truth 1.5 kg boxes. Each box's outcome uses Run 1's matching
  (class-agnostic greedy, IoU 0.5) at that run's threshold. There are four possible outcomes:
  `1.5kg`, `0.5kg`, `missed`, `other`.
- **M** = boxes read as `0.5kg`. **K** = boxes read as `1.5kg`. For Run 1, M 40 and K 36
  (21 missed, 3 other).
- Paired test, since both runs see the same 100 boxes. b = boxes misread by 2C and not by 2A;
  c = boxes misread by 2A and not by 2C. The one-sided exact McNemar test gives
  p = P(X ≤ c) for X ~ Bin(b + c, ½).

**6. Guardrails**, 2A against 2C:
- **G1:** test mAP@50–95 (`rfdetr`'s own number, the one that gave Run 1 its 0.741):
  2A ≥ 2C − 0.02.
- **G2:** val mAP@50–95 of the selected checkpoint: 2A ≥ 2C − 0.02. Val shares dates with
  train, so this checks in-distribution quality.
- **G3:** per-class test AP@50–95 for every class except 0.5 and 1.5 kg: none falls more than
  0.05 below 2C.
- The margins are judgement, fixed before training, because no variance estimate exists (one
  seed per arm). 0.02 is about 3 % of Run 1's 0.741. 0.05 is the bottom of the 0.04–0.08
  val → test drop that 9 of 11 classes already show.

**7. Decision rule**, applied in this order:
1. **Void, the failure doesn't reproduce:** M(2C) ≤ 20. The control already halves the
   misreads with no colour augmentation, so the Run 1 failure isn't stable across retrains.
   That leaves no claim about colour either way. Run 1 vs 2C is reported.
2. **Fixed:** all of
   - (a) M(2A) ≤ ⌊M(2C) / 2⌋, i.e. at least half of the control's misreads are gone;
   - (b) McNemar p < 0.05;
   - (c) the misreads became correct reads, not misses: K(2A) − K(2C) ≥ ½ (M(2C) − M(2A));
   - (d) G1, G2 and G3 all hold.
3. **Fixed at a cost:** (a)–(c) hold, but a guardrail fails.
4. **Displaced:** (a) and (b) hold, but (c) fails. Misreads turned into misses, the pattern
   the recolouring test's ring arm showed (28 of 40 missed).
5. **Not fixed:** anything else.

**8. Secondary**, reported but not deciding anything:
- M and K split by backdrop side, using Run 1's groups (`backdrop_boxes.csv`, ring b\* > 0
  vs ≤ 0). The side belongs to the image, so it carries over to any run. If the fix is real,
  it should show on the yellow side.
- 0.5 kg test precision (Run 1: 0.715) and the full test confusion table for each run.
- Test mAP@50 and the val → test gap for each run.
- Every number above for Run 1 vs 2C: the only available measure of how much a retrain drifts.

**What each answer means.**
- *Fixed*: 2A becomes the shipped checkpoint (HF weights and the demo). The README tells the
  whole chain: Run 1 → error analysis → association → intervention → fix.
- *Fixed at a cost*, *displaced* or *not fixed*: Run 1 stays the shipped checkpoint. The
  README reports Run 2 with its numbers, as a negative or mixed result.
- *Void*: Run 1 stays. The README says the 1.5 → 0.5 kg failure didn't reproduce in a
  retrain, so part of its size is run-to-run noise.

**What this can't show.**
- **The test date is no longer untouched.** Its errors chose the question, the backdrop check
  and the recolouring test measured its images, and now the fix is scored on it. 2A's test
  numbers answer "does it fix the known failure on this date", not "how well does it do on a
  new date". The README must say so. A clean generalisation number needs a date nobody has
  looked at, and this dataset has none left.
- One seed per arm. McNemar covers which boxes happen to be in the test set, not training
  randomness. 2C vs Run 1 is the only look at the second.
- The augmentation simulates the colour of the light. A fix supports "robustness to scene
  colour helps". It doesn't show that white balance or illumination was the real cause on
  2025-05-04. The recolouring test's caveat stands.
- `PlanckianJitter` changes plate and backdrop together. The recolouring test changed only
  what surrounds the plate. The augmentation is the realistic intervention and the recolour
  the controlled one, and they are not the same experiment.

**Order:** this protocol → the notebook change (a 2C/2A switch; pinned backend, seed and
`albumentations`; the config check), plus a local test that both `aug_config`s build through
`rfdetr` and that 2A's transform leaves boxes alone and moves chroma more than lightness on a
synthetic image → both Kaggle runs on one commit → evaluation of both, output committed
unedited. Nothing above gets edited once training starts. Anything learned goes in the log as
a separate, dated entry.

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

### 2026-09-25 — backdrop check run (protocol committed first, `3b89165`)
- **Order in history:** protocol `3b89165` → code + synthetic tests `c72e7f5` → one run,
  output committed unedited `e11edfe`. `uv run custom-detector backdrop` reproduces it.
- **Pre-registered verdict: for.** AUC 0.715, cluster-bootstrap 95% CI 0.574–0.848 (0 resamples
  skipped). It only just clears the bar (≥ 0.70, lower bound > 0.55). Ring b\* median:
  misread 45.2 (IQR 43.0–47.4) vs correct −4.7 (IQR −12.4 to 48.3). No box was unmeasurable.
- **Protocol slip, no effect on the answer:** the question said 97 test 1.5 kg boxes. There
  are 100. The other 3 are other-class confusions, which the protocol already puts outside the
  M-vs-C comparison.
- **The within-image control was empty.** Every test photo holds exactly one 1.5 kg plate, so
  there are no same-photo M/C pairs, and backdrop can't be separated from "this photo". That
  could have been checked from the labels alone before pre-registering. Lesson: count the
  boxes per cluster before designing a within-cluster control.
- **Exploratory (cut-off picked after seeing the IQRs, so not evidence at the protocol's
  level):** with "yellow backdrop" = ring b\* > 30, all 40 misreads are on yellow. Plates on
  yellow: 12 correct / 40 misread / 15 missed. Plates on anything else: 24 / 0 / 6. The AUC
  understates this because the correct group is bimodal: a third of the correctly read plates
  are on yellow too. The hours overlap (misread 19–21 h, correct 18–21 h), so it isn't simply
  one block of the evening. Yellow backdrop looks close to *necessary* for the misread, but
  not sufficient.
- **Secondary, plate chroma C\*:** misread 38.9 vs correct 41.9 (IQRs overlap). So "pale"
  holds only weakly.
- **What it means:** an association on one date, not a cause (see the protocol's caveat). Run 2
  now has a named failure mode. Before training, two cheaper options to weigh:
  (a) **test-time intervention:** recolour only the ring pixels of misread plates (yellow → the
  dark blue) and re-run inference with the Run 1 checkpoint on CPU. If the reads flip to
  1.5 kg, that's causal evidence with no training at all. (b) Run 2 with hue/saturation
  augmentation, success criterion written first: test 1.5 kg recall on yellow-backdrop plates
  and 0.5 kg precision, with no other class losing more than its Run 1 CI.

### 2026-09-30 — recolouring test run (protocol committed first, `41733e3`)
- **Order in history:** protocol `41733e3` → transform + synthetic tests `89ac34e` → gate +
  arm code + synthetic tests `c0defd5` → one run, output committed unedited. `rfdetr` pinned
  to **1.11.0** (Run 1's own version — no version gap to explain away). Checkpoint SHA-256
  `822a971f`; `torch 2.14.0+cpu`; `uv run custom-detector recolour` reproduces it.
- **Baseline gate: passed, 0/76 dropped.** The local CPU pipeline reproduced Run 1's outcome
  on every one of the 76 unaltered arm boxes before a single pixel was touched.
- **Pre-registered verdict: for (scene).** Ring flips 2/40 (5.0%, CI 1.4–16.5%) — nowhere near
  the ≥ 20/40 bar. Whole flips 24/40 (60.0%, CI 44.6–73.7%) — clears its ≥ 20/40 bar easily.
  Sham flips 0/40, so this isn't pipeline noise. Recolouring the entire scene blue fixes most
  misreads; recolouring only the ring around the plate mostly doesn't.
- **What the ring arm actually did, not just the flip count:** 28 of the 40 ring boxes came
  back `missed` (no detection at all), 10 stayed `0.5kg`, only 2 flipped to `1.5kg`. A local
  colour patch right at the plate's edge isn't read as "a blue backdrop" — it reads as an
  artifact the model doesn't recognise, so it drops the box rather than reclassifying it. The
  whole arm also produced 16 `missed` (a whole-image colour shift is still off-distribution),
  but 24 `1.5kg` against only 0 `0.5kg` — when it doesn't confuse the detector outright, it
  reads the plate correctly, not merely "differently."
- **Reverse (C-blue → yellow ring), not decisive:** 6/19 flipped to `0.5kg` (31.6%, CI
  15.4–54.0%), 12 `missed`, 1 stayed `1.5kg`. Median ring clipping **73.3%** — most of these
  ring pixels can't actually reach the yellow target from a dark-blue starting point within
  the sRGB gamut, so "yellow ring" here is a best-effort, heavily clipped approximation, not a
  clean recolour. Any reading of Reverse should note that up front.
- **Keep (C-yellow ring → blue), not decisive:** 14/17 stayed `1.5kg` (82.4%, CI 59.0–93.8%),
  3 `missed`. Recolouring an already-correct read's ring mostly doesn't break it.
- **Median confidence change (secondary, IoU ≥ 0.5 to the box, any confidence):** ring +0.293,
  whole +0.678, sham +0.000 (round trip is a genuine no-op), reverse −0.615, keep −0.014.
  Consistent with the flip counts — Whole moves confidence up a lot, Ring a little, Reverse
  down a lot.
- **What it means:** the backdrop *causes* the misread with this checkpoint, but through the
  colour of the whole scene, not through a narrow strip touching the plate — closer to the
  camera's colour balance or overall lighting than to "what's directly behind this plate."
  Run 2's success criterion, if it happens, should target scene-level colour augmentation
  (global hue/saturation jitter), not an edge/context-only augmentation. Per the protocol,
  this gives Run 2 a named purpose; also per the protocol's own limits, this is one checkpoint,
  one test date, and Whole's own 16 `missed` boxes mean the scene-colour route isn't fully
  reliable either. Move next to weighing Run 2 against shipping the demo with this finding
  reported as known.

### 2026-10-04 — Run 2 notebook switch built (protocol `93f0e22`, implementation `4b09e46`, nothing trained)
- **What was built:** `src/detector/run2.py` (the two arms, a strict transform build, the config
  check), the notebook's `RUN = "2C" | "2A"` switch, and `tests/test_run2.py`. No protocol text
  changed. Two findings from building it are recorded here as reproducibility facts, not as
  anything learned from a result: no Run 2 number exists yet.
- **`rfdetr` 1.11.0 builds user augmentation leniently.** A transform it cannot build (a
  misspelt name, a bad parameter) is logged as a warning and skipped, and training goes ahead.
  A typo in `PlanckianJitter` would therefore have trained a second control and looked
  normal. The arm's transforms are now built with `strict=True` and checked by name before
  training, and a test shows a misspelt transform raises. The same
  call with `strict=False` returns the surviving transforms without an error.
- **`albumentations` is not installed by `rfdetr[train]`.** It sits in the separate `augment`
  extra. Run 1's install (`rfdetr[train]`) therefore had no `albumentations`, so its
  `augmentation_backend="cpu"` setting resolved to torchvision, which is what the protocol's
  confound argument assumed. Both Run 2 arms install `albumentations==2.0.8` explicitly,
  assert the installed version, and pin `augmentation_backend="albumentations"`. The pin is
  also in the dev dependency group, so CI exercises the same tests.
- **Config check, as implemented.** Against Run 1's saved `training_config.json`, everything
  must match except `aug_config`, `augmentation_backend`, `seed`, `dataset_dir`, `output_dir`
  and the directory part of `model_config.pretrain_weights`. The weights file name must still
  match. The cache directory is a property of the Kaggle image, not of the experiment, so a
  different one does not void a run. The check runs before training on the planned config and
  again on the saved one, and the result is stored in `run_record.json`.
- **Local test result:** both arms build through `rfdetr` and take the Albumentations
  training path; on a neutral grey image `PlanckianJitter` leaves boxes and labels untouched,
  moves L\* by at most 2.2 and chroma by up to 34.9 (mean 20.4 against 0.7 for L\*) over 60
  draws, in line with the protocol's earlier measurement.
- **Launch discipline.** Both arms run on one pinned commit SHA, 2C first, then 2A. 2C's test
  results are not looked at before 2A is launched. The Run 2 evaluator is written and
  committed before either run's outputs are opened.

### 2026-10-04 — Run 2 evaluator written before any Run 2 output exists
- **What:** `src/detector/run2_eval.py`, run as `custom-detector run2-eval`. It implements
  sections 4-8 of the Run 2 protocol and nothing else: per-run threshold by Run 1's rule,
  the outcome of each of the 100 test 1.5 kg boxes (`1.5kg`, `0.5kg`, `missed`, `other`), M
  and K, the paired one-sided exact McNemar test, G1-G3, the decision rule in its stated
  order (void, fixed, fixed at a cost, displaced, not fixed), and the secondary tables.
  Tests cover every state on synthetic predictions, the boundaries of every criterion, and
  the refusal cases below. No Run 2 file has been opened.
- **Run 1 through the same code.** Scored by this evaluator, Run 1 gives threshold 0.80, K 36,
  M 40, missed 21, other 3, 0.5 kg precision 0.715, test mAP@50-95 0.741, with all 40
  misreads on the yellow side. These match the numbers recorded before Run 2 existed, and the
  evaluator re-checks this on every run and refuses to score if it ever stops being true.
- **Two readings the protocol left open, fixed now.** (1) G2's "val mAP@50-95 of the selected
  checkpoint" is the best value over epochs of `val/mAP_50_95` and `val/ema_mAP_50_95`, since
  `rfdetr` selects `checkpoint_best_total` on that metric. For Run 1 that is 0.8293 at epoch
  36. (2) A guardrail holds when the drop equals its margin exactly (-0.02 for G1 and G2, -0.05
  for G3). `test/AP/<class>` in `metrics.csv` is AP@50-95 (its mean over the 11 classes equals
  `test/mAP_50_95`), so G3 uses it directly.
- **Refusals, so the rule cannot run on the wrong thing.** Nothing is scored unless both arms'
  five files are present. A run whose saved config differs from Run 1's outside the allowed
  fields, whose record names another arm or a failed notebook check, or whose two records
  differ in commit, `rfdetr` or `albumentations` version, split sizes, GPU or prediction
  threshold, gets the verdict `invalid` (numbers still shown, no decision). The test set
  must hold exactly 100 1.5 kg boxes.
- **Order from here:** launch 2C, then 2A, both at `4b09e46`; download both; run
  `custom-detector run2-eval`; commit the output unedited; accept the verdict it gives.

### 2026-10-04 — Run 2 result: verdict *fixed* (raw outputs and evaluator output `26864ee`, unedited)
- **Order in history:** protocol `93f0e22` → notebook switch `4b09e46` → evaluator `b89c36e` →
  both arms trained on `4b09e46`, outputs committed unedited as `26864ee`. The verdict below is
  the one `results/run2_evaluation.json` printed. Nothing about it was chosen after the numbers.
- **Validity.** Config check ok for both arms (nothing differs from Run 1 outside `aug_config`,
  `augmentation_backend`, `seed` and paths). Same commit, `rfdetr` 1.11.0, `albumentations`
  2.0.8, Tesla T4, split 1,263 / 331 / 402 for both. `invalid_reasons` is empty. One fact the
  config check does not cover: both Run 2 arms trained on `torch 2.11.0+cu128`, Run 1 on
  `2.10.0+cu128`. It cannot affect 2A vs 2C (same on both sides). It is one more reason to treat
  Run 1 as a reference and not as a third arm. Training took 125.6 min (2C) and 137.2 min (2A);
  best val epoch 18 and 21 (Run 1: 36).
- **Primary result, the 100 test 1.5 kg boxes** (each run at its own val-chosen threshold):

  | run | threshold | read 1.5 kg (K) | read 0.5 kg (M) | missed | other |
  |---|--:|--:|--:|--:|--:|
  | Run 1 (reference) | 0.80 | 36 | 40 | 21 | 3 |
  | 2C control | 0.67 | 40 | 51 | 6 | 3 |
  | 2A colour | 0.53 | 72 | 25 | 0 | 3 |

- **Decision rule, in the order written.**
  - *Void* needed M(2C) ≤ 20. M(2C) is 51, so the failure reproduced in a retrain. Not void.
  - (a) M(2A) ≤ ⌊51 / 2⌋ = 25: **25. Holds with zero slack.** One more misread in 2A and (a)
    fails, the rule falls through to *not fixed*. The count went 51 → 25, so 26 of 51 (51%)
    are gone.
  - (b) McNemar, 2A vs 2C: b = 26 boxes misread by 2C only, c = 0 misread by 2A only,
    one-sided exact p = 0.5^26 = 1.5e-8. Every box 2A still misreads, 2C misreads too.
  - (c) K(2A) − K(2C) = 32 ≥ (51 − 25) / 2 = 13. The misreads became correct reads: 2A has 0
    missed (2C 6), so this is not the *displaced* pattern.
  - (d) G1 test mAP@50–95 0.762 vs 0.748 (margin −0.02); G2 val 0.821 vs 0.823 (margin −0.02);
    G3 no class below 2C by more than 0.05, the largest drop is 2 kg at 0.011. All hold.
  - **Verdict: fixed.** Consequence per the protocol: 2A is the shipped checkpoint.
- **Secondary (decides nothing).** All numbers from `run2_evaluation.json`.
  - By backdrop side (Run 1's groups): yellow side, M 51 → 25, K 17 → 49, missed 6 → 0. Blue
    side, M 0 → 0 and K 23 → 23. The change sits where the diagnosis said it should. All 25
    remaining misreads are on the yellow side.
  - 0.5 kg test precision 0.715 (Run 1) / 0.660 (2C) / 0.769 (2A). Test mAP@50 0.953 / 0.960 /
    0.981. Val → test mAP@50–95 gap 0.088 / 0.075 / 0.059.
  - Test AP@50–95 of the two target classes, which G3 excludes: 1.5 kg 0.611 / 0.632 / 0.702,
    0.5 kg 0.516 / 0.594 / 0.671.
- **Not pre-registered, read off the same files. Observations, not tests.**
  - *Retrain drift is large on this metric.* Run 1 → 2C, same recipe on a different backend and
    seed: M 40 → 51, missed 21 → 6. 2C moved M by 11 and missed by 15 with no colour change at
    all. 2A's drop of 26 against 2C is more than twice that drift, but it is one sample of
    drift, and it is the reason the single seed matters. McNemar says the 26 is not a
    test-set-sampling accident. It cannot say 2A would beat 2C on a second seed.
  - *The runs used different thresholds* (0.80, 0.67, 0.53). Each is the value the pre-registered
    micro-F1 rule picked on that run's own val, so the threshold step is part of each model's
    frozen evaluation and is not a deviation. At 0.53, 2A has 12 predictions matching no
    ground-truth box (0.5 kg ×4, 2 kg ×3, 1.5 kg ×2, collar ×2, 5 kg ×1); Run 1 and 2C have none.
    That is context only. The frozen experiment does not establish how much of the result, if any,
    comes from the threshold, and this entry does not claim any. An equal-threshold comparison
    was not pre-registered and was not performed.
  - 2A is not a cure: 25 of 100 test 1.5 kg plates are still read as 0.5 kg.
- **What the result supports.** On the date the failure was found, training with a global
  colour-temperature cast removes about half the 1.5 → 0.5 kg misreads that the same recipe
  without it leaves, at no measurable cost to the other classes or to overall mAP. The whole
  chain holds together: Run 1 error analysis → backdrop association → whole-scene recolour test
  → colour-cast augmentation → fewer misreads, concentrated on the yellow side.
- **What it does not show. These stay in every document that quotes the result.**
  1. **The test date was used for diagnosis.** Its errors chose the question, the backdrop check
     and the recolouring test measured its images, and now the fix is scored on it. 2A's test
     numbers say whether the known failure is fixed on this date, not how the model does on a
     new one. A clean number needs a date nobody has looked at, and this dataset has none left.
     Run 1's 0.741 is the cleaner new-date estimate, because nothing was tuned against it.
  2. **One seed per arm.** McNemar covers which boxes are in the test set, not training
     randomness. Run 1 vs 2C is the only look at the second, and it moved by 11.
  3. **Fixing it does not show lighting was the physical cause.** The augmentation simulates
     the colour of the light. Succeeding at it supports "robustness to scene colour helps". It
     does not show that white balance or illumination caused the Run 1 failure on 2025-05-04. It
     could equally be a backdrop-and-plate colour combination that was rare in training, which
     any colour-varying augmentation would soften. Also, `PlanckianJitter` moves plate and
     backdrop together, while the recolouring test moved only the surroundings, so the two
     experiments are related but not the same one.
- **Consequences, as the protocol states them.** 2A is the shipped checkpoint: the HF weights
  and the demo come from `models/kaggle-run2a/run/`. Not done yet: the HF upload, the demo, the
  ship gate. No further training or tuning against this test date. Any new change needs its own
  protocol, and its test numbers would be another look at the same images.

### 2026-10-04 — 2A checkpoint published to the Hugging Face Hub

Publication only. No training, no re-scoring, and nothing in `results/` from Run 2 changed.

- **Where:** https://huggingface.co/Prithv122/custom-detector (public), Hub commit
  `be16623ab1de46cc2e2b9e4df477af5d71d04c58`. Two files went up in one commit:
  `checkpoint_best_total.pth` (from `models/kaggle-run2a/run/`) and the model card
  (`MODEL_CARD.md` here, `README.md` on the Hub).
- **The file:** 127,623,409 bytes, SHA-256
  `f802956130702712f05bfb82d8b437ea1648efe642ff90bbba2fb38a1d8bce1f`. The Hub's own LFS
  SHA-256 for the file is the same value.
- **The card** is written only from `results/run2_evaluation.*` and the Run 2 write-up above. It
  carries all three limits from that entry, Run 1's 0.741 as the new-day estimate, and the
  zero-slack pass of criterion (a). It makes no claim about colour or lighting robustness in
  general. Weights are released as Apache-2.0, following the RF-DETR-S base model, with CC BY 4.0
  attribution for the dataset.
- **Round trip** (`scripts/verify_hf_checkpoint.py`, output `results/hf_publication.json`): it
  downloads the file at that revision into an empty cache, so the local copy can't stand in for it.
  The downloaded SHA-256 matches. CPU inference with the downloaded copy (torch 2.14.0, rfdetr
  1.11.0) on the 100 test 1.5 kg boxes at 2A's threshold 0.53 gives 72 / 25 / 0 / 3 (read 1.5 kg
  / read 0.5 kg / missed / other). That is the same as the Kaggle GPU predictions, with **0 of 100
  box outcomes different**. This checks that the published file works. It is not a new
  measurement, because these are the same boxes Run 2 was scored on.
- Not done: the demo, README §7, the ship gate.

### 2026-10-04 — local demo on the published 2A checkpoint

A Gradio app, `custom-detector demo`, on the Hub checkpoint at commit `be16623`. No training,
no scoring, nothing in `results/` changed, and the demo produces no new accuracy number.

- **Default threshold 0.53, not a round 0.5.** 0.53 is 2A's val-chosen value, the one the model
  card, the round-trip check and every test number use. A slider covers other values, and the
  page says which one the numbers belong to.
- **One model call per photo.** The model runs at the evaluation floor (0.01) and the slider
  filters those detections with `>=`, the same cut the evaluation applies. Moving the slider
  costs a redraw, not an inference.
- **The checkpoint is pinned and hashed.** It is fetched at the pinned Hub commit and its SHA-256
  is checked before `torch.load` runs on the pickled file, also for a `--checkpoint` path. A test
  ties the pinned commit, hash and default threshold to `results/hf_publication.json` and
  `results/run2_evaluation.json`.
- **Local only.** It binds to 127.0.0.1; photos never leave the machine. Gradio is an optional
  `demo` extra, so the evaluation code and a plain `uv sync` don't carry it. CI installs it so the
  app wiring is tested.
- **No weight total.** The page lists detections and counts per class. It does not sum plate
  weights: the loaded-weight metric was dropped for lack of ground truth, and a total shown next
  to a photo would read as that measurement.
- **No example photos in the repo.** The kept images are third-party photos whose author is
  unverified (see the open question below), so `--examples <folder>` reads a local folder instead.
- **The limits on the page are quoted from the result files** (72 / 25 / 0 / 3 at 0.53, 51 to 25,
  12 predictions matching no labelled box), and a test pins the first two to
  `results/run2_evaluation.json`.
- **Found while checking it in a browser:** the slider's `release` event fires only when the handle
  is dragged, so typing a value in its number box changed nothing. Switched to `change`; typing 0.9
  then took one example photo from 4 detections to 2.
- **Review fixes (diff-only review, before this entry):** `launch` now passes `share=False`,
  because Gradio turns sharing on from an environment variable or a notebook host and the page says
  photos stay on this machine. A threshold moved while the model is still running is re-applied once
  inference ends (`photo.change(...).then(...)`), so the slider and the boxes can't disagree. Tests
  now pin the page's numbers to the result files and the slider's trigger. Left alone: the hash is
  checked and the file then read again, which only a local process with write access to the Hub
  cache could exploit.
- **Evidence:** 164 tests pass (22 new), ruff clean. The app launched from the Hub path (download,
  hash check, CPU load), and clicking a test-split example drew boxes, a count line and a table.
  That is a smoke check on one photo, not an evaluation.
- **QA pass (same day, after the review fixes):**
  - All 100 test-date 1.5 kg photos sent through the running app's HTTP endpoint at 0.53 and scored
    with the evaluation's own matching: 72 / 25 / 0 / 3, **0 of 100 outcomes different** from the
    Kaggle 2A predictions. This goes through the app's rounded output table, so it checks the app,
    not the model again (same boxes as Run 2).
  - Five hand-picked photos in the browser matched their recorded 2A outcome (two still read as
    0.5 kg, two fixed, one "other"). On a still-misread one the same box is 0.5 kg at 0.776 and 1.5 kg
    at 0.452, below the cut, which is the failure the page describes.
  - Odd inputs: grayscale, RGBA, WebP, a 4000x3000 JPEG (10.6 s), a 1x1 image and a blank one all
    return normally; a text file renamed `.png` is rejected by the upload component with an error
    toast. CPU time: median 3.9 s per photo, max 17.8 s on this machine.
  - A copy of the tracked files in a fresh environment with an empty Hugging Face cache and no
    `models/` folder: ruff clean, 162 passed and 2 skipped (the two tests that need the Roboflow
    export or the prepared dataset), and the app downloaded the pinned snapshot `be16623` and ran.
    Reloading the page returns to a clean state at 0.53.
- Not done: a hosted version (a public Space), README §7, the ship gate.

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
