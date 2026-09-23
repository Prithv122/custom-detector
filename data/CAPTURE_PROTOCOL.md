# Capture protocol

Each photo gets two independent ground truths:

1. **Bounding boxes**, drawn later during labelling.
2. **The actual load on the bar**, written into `data/loads.csv` *at the gym*, from what is
   physically on the bar — never copied from the schedule below and never read off the labels.

Keeping these apart is the whole point. The loaded-weight metric only means something if its
answer key was never derived from the boxes.

---

## Before the first visit

- [ ] Ask gym staff for permission to photograph equipment.
- [ ] Make slate cards `L01`–`L24` plus `LOOSE` (paper, or big text on a second screen).
- [ ] Phone: rear main camera, no portrait/beauty mode, lens wiped. Note the phone model.
- [ ] Bring `data/loads.csv` on the phone or on paper.

## At the gym — inventory first (≈5 min)

Record in the `notes` of the first `loads.csv` row:

- [ ] Plate type (bumper / iron / rubber-coated), brand, colour of each denomination.
- [ ] How many of each denomination are available. **The full schedule needs 4 of each.**
- [ ] Any plates outside the six classes (1.25 kg, 0.5 kg, lb plates) — these never go on the bar.
- [ ] Bar type and its stamped weight. Collar type and weight.
- [ ] One face-on photo of each denomination lying flat — reference only, not dataset.

## Rules for every bar photo

- [ ] Loaded **symmetrically**, heaviest plate innermost.
- [ ] Only the six classes on the bar.
- [ ] No collars, or light spring clips (recorded). **No 2.5 kg competition collars** — they
      would break the totals and look like plates.
- [ ] Camera at **~30–45° to the sleeve**: rims and part of each face visible.
      Not end-on, not square side-on.
- [ ] **No people, no faces, no mirrors showing people.**
- [ ] **Slate first:** before each load, photograph its card. Photos are sorted by these later.
- [ ] **6 photos per load:**
      3 framing the near sleeve only, 3 framing both sleeves; vary distance (≈1 m / ≈2 m)
      and height (standing / crouching).

## Load schedule — Gym A

Per side, innermost first. Totals assume a 20 kg bar and no collars; **the recorded total is
whatever was actually loaded.** If a load can't be built, substitute and write down what was used.

| Load | Per side | Total | Split |
|---|---|---|---|
| L01 | 25 | 70 | train |
| L02 | 20 | 60 | train |
| L03 | 15 | 50 | train |
| L04 | 10 | 40 | train |
| L05 | 5 | 30 | train |
| L06 | 2.5 | 25 | train |
| L07 | 25 + 2.5 | 75 | train |
| L08 | 20 + 5 | 70 | **val** |
| L09 | 15 + 10 | 70 | **test** |
| L10 | 10 + 5 | 50 | train |
| L11 | 5 + 2.5 | 35 | train |
| L12 | 25 + 20 | 110 | train |
| L13 | 20 + 15 | 90 | train |
| L14 | 25 + 10 | 90 | **val** |
| L15 | 15 + 5 + 2.5 | 65 | **val** |
| L16 | 20 + 10 + 5 | 90 | **test** |
| L17 | 25 + 15 + 2.5 | 105 | **test** |
| L18 | 10 + 10 | 60 | train |
| L19 | 5 + 5 + 2.5 | 45 | train |
| L20 | 20 + 20 + 5 | 110 | train |
| L21 | 25 + 25 | 120 | train |
| L22 | 15 + 15 + 10 | 100 | train |
| L23 | 25 + 20 + 10 + 5 + 2.5 | 145 | train |
| L24 | 2.5 + 2.5 | 30 | train |

The split is fixed **now, before any training**: 18 train / 3 val / 3 test loads, and both val
and test contain every class. Several loads share a total (70, 90 and 110 kg each appear more
than once) on purpose, so a right total with wrong plates is visible.

## Loose-plate shots — detection training only

- [ ] Slate `LOOSE`, then ~30 photos: plates on the storage tree, leaning, stacked, on the
      floor, face-on and angled. These have no load ground truth. They go to **train only**
      and are excluded from the weight metric.

## Gym B — generalisation test only

Same rules, 5 photos per load, loads **L01–L06, L12, L15, L20, L23** (every class, singles
and stacks). Gym B photos are **never** used for training, validation or tuning.

## Expected volume

| Set | Photos |
|---|---|
| Gym A bar photos (24 loads × 6) | 144 |
| Gym A loose plates | ~30 |
| Gym B bar photos (10 loads × 5) | 50 |
| **Total** | **~225** |

About 60–75 minutes at Gym A, most of it spent changing plates.

## After the visit

- Copy photos to `data/raw/gym_a/` or `data/raw/gym_b/` (git-ignored — images are published
  as a dataset, not committed).
- Check every `loads.csv` row: `total_kg = bar_kg + 2 × sum(side_plates_kg) + 2 × collar_kg_each`.
