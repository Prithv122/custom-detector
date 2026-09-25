Test 1.5kg boxes: 100 · confidence threshold 0.8 (Run 1's, from val)

| group | measurable boxes | images | unmeasurable | ring b* median (IQR) | plate C* median (IQR) |
|---|--:|--:|--:|--:|--:|
| correct | 36 | 36 | 0 | -4.7 (-12.4 to 48.3) | 41.9 (40.0 to 45.2) |
| misread | 40 | 40 | 0 | 45.2 (43.0 to 47.4) | 38.9 (37.5 to 42.0) |
| missed | 21 | 21 | 0 | 48.4 (11.5 to 50.0) | 40.7 (38.6 to 42.3) |
| other | 3 | 3 | 0 | 43.6 (13.3 to 43.9) | 21.3 (17.1 to 28.7) |

AUC (ring b*, misread > correct): **0.715** · 95% CI 0.574 to 0.848 (cluster bootstrap over images, 10000 reps, seed 20260925, 0 skipped)
Within-image: 0 photos with both groups, 0 pairs, AUC -

**Verdict: for**
