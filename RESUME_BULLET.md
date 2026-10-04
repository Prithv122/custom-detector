# Resume Bullets — Custom Object Detector

Form: **action → technical specifics → measured outcome.** Numbers or it doesn't go on the resume.

---

## Bullets

- Audited a 2,251-image public weightlifting-plate dataset with 256-bit perceptual hashing, found 315 images with a near-duplicate in another split, excluded 255 images that couldn't be licensed, and re-split by capture date (1,996 images, 11 classes) so the test set is a whole day the model never saw.
- Fine-tuned RF-DETR-S on a Kaggle T4 to 0.953 mAP@50 / 0.741 mAP@50–95 on that held-out day, then traced its worst failure (40 of 100 test 1.5 kg plates read as 0.5 kg) to scene colour with two analyses whose decision rules were committed before any number was computed.
- Cut that misread from 51 to 25 of 100 against a matched control retrain by adding a colour-temperature augmentation (paired exact McNemar p = 1.5e-8, single seed per arm), with no other class losing more than 0.011 AP. Scored on the day used for diagnosis, so it is a targeted fix and not a new-day estimate.

**Shorter form, one line:** Fine-tuned RF-DETR-S on a leakage-free, date-held-out split (0.741 mAP@50–95, 0.953 mAP@50) and cut a diagnosed 1.5 kg → 0.5 kg misread from 51 to 25 of 100 against a matched control (single seed, diagnosed date).

## Which roles this supports

- [x] Data Scientist / ML
- [x] AI Engineer (LLM/NLP/CV)
- [ ] Data Engineer
- [ ] Data Analyst / Python Developer

The dataset audit and committed manifest touch data engineering, but nothing here is a pipeline or
a service, so those two roles are left unchecked.

## Keywords this project earns

_Only list what you actually used and could be questioned on._

RF-DETR · object detection · mAP@50 / mAP@50–95 · COCO evaluation · fine-tuning on a Kaggle T4 ·
perceptual hashing (train/test leakage) · date-grouped split · pre-registered analysis ·
CIELAB colour space · data augmentation (`PlanckianJitter`, Albumentations) · matched control
experiment · paired exact McNemar test · cluster bootstrap

## Before using a number

- **0.741 / 0.953** are Run 1's, scored on an untouched day. **0.762 / 0.981** are the shipped
  colour-augmented model's, scored on the day used for diagnosis. Don't quote the second pair as
  how the model does on a new day.
- "Cut misreads from 51 to 25" is relative to the control retrain (2C), not to Run 1 (40).
- Always say single seed per arm. The 25 met the pre-registered bar exactly, with no slack.
- Don't write that lighting caused the failure. The result shows scene colour matters to the
  model, not what produced that colour in the real scene.

---

### Bad vs good

❌ "Built a machine learning model to predict customer churn using Python."
✅ "Built a churn classifier on 240k accounts (LightGBM, 1:40 class imbalance) with isotonic calibration and cost-sensitive thresholding, lifting precision@10% from 0.31 to 0.58 over the business's existing rules baseline."

The second one is answerable in an interview. The first invites the question you can't answer.
