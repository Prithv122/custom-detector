---
license: apache-2.0
library_name: rfdetr
pipeline_tag: object-detection
tags:
  - object-detection
  - rf-detr
  - computer-vision
---

# Weightlifting-plate detector (RF-DETR-S, Run 2A)

An 11-class object detector for weightlifting plates in side-on photos of a loaded barbell
sleeve: the ten plate weights from 0.5 kg to 25 kg, plus the collar. It is RF-DETR-S fine-tuned
with `rfdetr` 1.11.0. This is the **Run 2A** checkpoint, the one the project ships.

Code, dataset audit, pre-registered protocols and every number on this page:
https://github.com/Prithv122/custom-detector

## Files

| File | Size (bytes) | SHA-256 |
|---|--:|---|
| `checkpoint_best_total.pth` | 127,623,409 | `f802956130702712f05bfb82d8b437ea1648efe642ff90bbba2fb38a1d8bce1f` |

## How to use

```python
from huggingface_hub import hf_hub_download
from PIL import Image
from rfdetr import RFDETR

path = hf_hub_download("Prithv122/custom-detector", "checkpoint_best_total.pth")
model = RFDETR.from_checkpoint(path, trust_checkpoint=True, device="cpu")
det = model.predict(Image.open("sleeve.jpg").convert("RGB"), threshold=0.53)
for box, conf, name in zip(det.xyxy, det.confidence, det.data["class_name"]):
    print(name, round(float(conf), 3), box.round(1))
```

`trust_checkpoint=True` loads a pickled PyTorch file, so only load it from a source you trust.
Pin a `revision` to get the same bytes every time. The 0.53 confidence threshold is the one the
project picked for this checkpoint on the validation split, with a rule fixed before training.

## Model

| | |
|---|---|
| Architecture | RF-DETR-S (`RFDETRSmall`: DINOv2 windowed-small encoder), input resolution 512 |
| Library | `rfdetr` 1.11.0 (Apache-2.0) |
| Classes (11) | `0.5kg` `1kg` `1.5kg` `2kg` `2.5kg` `5kg` `10kg` `15kg` `20kg` `25kg` `zacisk` (collar) |
| Training | Tesla T4 on Kaggle, 137 min, torch 2.11.0+cu128. 50-epoch budget, early stopping (patience 10), batch 8 × 2 gradient-accumulation steps, lr 1e-4, seed 20261001. Best checkpoint selected on val by `rfdetr` |
| Augmentation | horizontal flip (p = 0.5) + `PlanckianJitter` (CIE D series, 4000–15000 K, p = 0.5), albumentations 2.0.8 |
| Training code | commit `4b09e46` of the GitHub repo |

## Training data

[*Weightlifting Plates*](https://universe.roboflow.com/projektciezary/weightlifting-plates) by
ProjektCiezary, Roboflow Universe, version 10, **CC BY 4.0**. Changes made: 255 of 2,251 images
were removed (third-party competition-video crops and press photos, unlabelled images,
contradictory labels), leaving 1,996. The published split leaked near-duplicate photos across
splits, so it was replaced: train 1,263, val 331 (whole capture sessions), test 402 (one whole
capture date, 2025-05-04, never used for training). The images are side-on phone photos of one
loaded sleeve, mostly coloured bumper plates on a loading rig, taken on 7 dates between May and
November 2025.

## Why this checkpoint

Run 1 (no colour augmentation) misread 40 of the 100 test-date 1.5 kg plates as 0.5 kg. Those
misreads all sat in front of a yellow backdrop, and recolouring the whole scene around them
flipped 24 of the 40 to a correct read. Run 2 then trained two arms on one commit, with one seed
each, and scored them by a rule written before training:

- **2C (control):** horizontal flip only.
- **2A (shipped):** horizontal flip plus a colour-temperature cast (`PlanckianJitter`).

Results on the 100 test 1.5 kg boxes (each arm at its own val-chosen threshold):

| Run | Threshold | Read as 1.5 kg (correct) | Read as 0.5 kg (the known failure) | Missed | Other |
|---|--:|--:|--:|--:|--:|
| Run 1 (reference) | 0.80 | 36 | 40 | 21 | 3 |
| 2C control | 0.67 | 40 | 51 | 6 | 3 |
| **2A shipped** | 0.53 | **72** | **25** | 0 | 3 |

- **Known failure, 1.5 kg read as 0.5 kg:** 51/100 (2C) → **25/100** (2A).
- **Correct 1.5 kg reads:** 40/100 (2C) → **72/100** (2A).
- **Test mAP@50–95:** 0.748 (2C) → **0.762** (2A). Val mAP@50–95: 0.823 → 0.821. No other
  class's test AP@50–95 dropped by more than 0.011 (the limit was 0.05).
- **Pre-registered verdict: fixed.** The success criterion was that 2A misreads at most half as
  many as 2C, rounded down: at most 25. 2A misread exactly 25, so **the criterion was met
  exactly at the boundary**. One more misread and the verdict would have been *not fixed*.
- Paired on the same 100 boxes, 26 were misread by 2C only and none by 2A only (one-sided exact
  McNemar, p = 1.5e-8).

## Limitations

Read these before quoting any number above.

- **The test date had already been used for diagnosis.** Its errors chose the question, two
  analyses measured its images, and the fix is scored on it. Run 2's numbers show that the known
  failure is reduced on that date, not how the model does on a new day.
- **Run 1's untouched-date test mAP@50–95 of 0.741 remains the clean new-day generalisation
  estimate.** It was measured before any of the diagnosis. The dataset has no untouched date
  left, so this checkpoint has no clean new-day number of its own.
- **One seed per arm.** McNemar's test covers which boxes happen to be in the test set, not
  training randomness. Retraining without the colour change moved the misread count from 40
  (Run 1) to 51 (2C), which gives a sense of how much one retrain can move it.
- **Augmentation success does not prove lighting caused the original failure.** The cast
  simulates the colour of the light and shifts plate and backdrop together. That a cast helps
  supports "robustness to scene colour helps here". It does not show that white balance or
  illumination caused the failure on that day.
- **This is not a general colour or lighting robustness claim.** The evidence is one failure
  mode on one capture date. 25 of 100 test 1.5 kg plates are still read as 0.5 kg, all of them
  in front of the yellow backdrop.
- **The arms used different confidence thresholds** (2C 0.67, 2A 0.53), each picked on that
  arm's own val split by a rule fixed before training. At 0.53, 2A makes 12 test predictions that
  match no labelled box (2C makes none). No equal-threshold comparison was pre-registered or run.
- **Narrow domain.** Side-on close-ups of one sleeve, from one dataset's phone photos on 7 dates.
  It has not been tested on other gyms, plate brands, camera angles or full-bar shots.
- Val shares capture dates with train, so val scores are optimistic by design.

## Licence and attribution

The weights are released under Apache-2.0, the licence of the RF-DETR-S base model. They were
trained on *Weightlifting Plates* by ProjektCiezary (Roboflow Universe, CC BY 4.0), filtered and
re-split as described above.
