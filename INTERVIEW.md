# Interview Prep — Custom Object Detector

**Five questions, five answers.** An unanswered question means this project is not shipped.

If you can't answer one, you don't understand that part of your own project yet — go back and understand it. This file is the difference between a portfolio that survives a technical screen and one that collapses in it.

---

### Q1. Walk me through the architecture in 90 seconds.

_A:_ The task is reading plate weights off a photo of a loaded barbell sleeve: 10 weights from
0.5 to 25 kg plus the collar, 11 classes. The data is a public Roboflow dataset. I audited it
before training: 2,251 images, of which I excluded 255 by rule (competition-video crops and press
photos the uploader can't license, plus unlabelled and contradictory ones), leaving 1,996. I also
replaced its split, because a 256-bit perceptual hash showed 315 images with a near-identical
twin on the other side. My split is by capture date and session: the test set is one whole date
the model never sees (402 images), val is whole sessions (331), train is the rest (1,263). Every
exclusion and split assignment is in one committed manifest, so the set rebuilds exactly.

The model is RF-DETR-S fine-tuned on a free Kaggle T4, two to three hours a run. I evaluate from
the saved predictions off the GPU, so the evaluation code has real tests. Run 1 gave test
mAP@50 0.953 and mAP@50–95 0.741. Its two worst classes, 0.5 and 1.5 kg, turned out to be one
error: 40 of 100 test 1.5 kg plates were called 0.5 kg. I traced that to scene colour with two
pre-registered checks, then trained a control and a colour-augmented model to test the fix.

### Q2. Why did you choose ___ over ___?

_A:_ The decision I'd defend most is **a global colour-temperature cast over hue/saturation
jitter** for the augmentation. In this domain the plate's colour is part of its label: 1.5 kg
is yellow, 0.5 kg is white, and they're close in size, so colour is how the model separates
them. A hue rotation turns a yellow plate green while the label stays 1.5 kg, which teaches the
model that colour doesn't matter. A colour-temperature cast (`PlanckianJitter`, CIE D series)
moves plate and background together, like a change in the light, so the plate's colour relative
to its scene survives. It also changes lightness by at most about 2.6 L\* while the b\* axis
swings over 70 units, so it's a colour change and not a brightness change.

The second is **a control run over comparing to Run 1**. Any non-empty augmentation config
switches `rfdetr` to a different resize backend, and Run 1 had no seed. Run 1 vs a colour run
would change colour, backend and seed at once. So I trained a control with flip only on the new
backend, and the colour arm differs from it only by the one transform.

The smaller ones: RF-DETR-S over Ultralytics YOLO because YOLO is AGPL-3.0 and RF-DETR-S is
Apache-2.0; a whole-date test set over the published random split because the random split leaks.

### Q3. What's the weakest part of this, and what would break first under load?

_A:_ Three things, in the order I'd worry about them.

1. **I have no clean number for the model I'm shipping.** The test date's errors chose the
   question and the date was then measured twice, so the shipped checkpoint's test scores say
   whether the known failure is reduced on that date, not how it does on a new one. Run 1's 0.741
   was taken before any of that and is the better new-date estimate. The dataset has no untouched
   date left.
2. **One seed per arm, and a verdict that passed with no slack.** The rule needed at most 25
   misreads and the colour model had 25. The paired test is very strong (26 boxes fixed, none
   broken, p = 1.5e-8) but it only covers which boxes are in the test set, not training
   randomness. And the control moved 11 misreads away from Run 1 with no colour change at all.
3. **It's not a cure:** 25 of 100 test 1.5 kg plates are still read as 0.5 kg, all of them
   against the yellow backdrop.

"Under load" doesn't quite apply: this is a model, not a service, and I haven't measured
inference latency. What breaks first is a new gym or lighting, since all the data is one set of
photos from 7 dates, one date held out.

### Q4. How do you know it works? What did you measure, and against what baseline?

_A:_ Against a held-out capture date, with the comparison written down before I looked. The
headline is mAP@50 0.953 / mAP@50–95 0.741 for Run 1, and I recomputed mAP@50 from the saved
predictions as a cross-check (0.954). The baseline for the fix isn't Run 1, it's the control
(2C): on the 100 test 1.5 kg plates the control calls 51 of them 0.5 kg and the colour model
calls 25, with correct reads going from 40 to 72 and misses from 6 to 0. Paired exact McNemar
gives 26 fixed, 0 broken. The misreads fall on the yellow-backdrop photos and not on the blue ones.

I also set guardrails before training so the fix couldn't buy itself with other classes: test and
val mAP@50–95 within 0.02 of the control and no other class more than 0.05 below it. All held,
the worst class drop was 0.011.

What I'd say about the limits: one date that was used for diagnosis, and one seed per arm. Also,
the two arms used different confidence thresholds (0.67 and 0.53), each picked on its own val by a
rule fixed before training. That's part of the frozen evaluation, but I never ran an equal-threshold
comparison, so I don't claim anything about how much the threshold contributed.

### Q5. Your first hypothesis was wrong, and you picked the fix after looking at the test set. Isn't "fixed" circular?

_A:_ It would be if I claimed generalisation, and I don't. It's a targeted fix scored on the
date where the problem was found, and I say so next to the number. What I did to keep it honest
was fix the question, the arms, the guardrails and the decision rule in a committed file before
training, build the evaluator and test it on synthetic predictions before any Run 2 output
existed, and accept the verdict it printed, including that it cleared one criterion exactly at
the limit.

The first hypothesis was wrong in a way I could show: I thought 0.5 and 1.5 kg were confused with
their same-colour heavy partners (5 and 15 kg). The pre-registered count was 1 of 100 and 0 of
103. The real error was 1.5 kg being read as 0.5 kg, different colour, neighbouring size. I also
found my own decision rule had a blind spot, because it only counted recall-side errors, so it
returned "mixed" on a clear miss. I wrote the lesson down: a rule over recall-side errors can't
see a precision-side failure.

The other thing I can't claim is cause. The colour augmentation working supports "robustness to
scene colour helps". It doesn't show that lighting or white balance caused the original failure.
The recolouring test, where I changed only the backdrop pixels and re-ran the same model, got 24
of 40 misreads to flip when I recoloured the whole scene and 2 of 40 when I recoloured only a
ring next to the plate, so the model's read depends on scene colour. What in the real scene
produced that colour I can't tell.

---

## 30-second pitch

I built a detector that reads weightlifting-plate weights off barbell photos, on a public dataset
that I audited first, because its published split leaked near-duplicate photos across train and
test. I re-split by capture date, trained RF-DETR-S, and found that one class pair failed on the
unseen day: 40 of 100 1.5 kg plates were read as 0.5 kg. Two pre-registered checks tied it to the
scene's colour, and a colour-cast augmentation, scored against a matched control, cut the
misreads from 51 to 25 of 100 without hurting any other class. The caveats are in the repo next
to the numbers: the test date was used for diagnosis, each arm has one seed, and I haven't shown
lighting was the physical cause.
