**Verdict: fixed** - 2A becomes the shipped checkpoint (HF weights and the demo).

### Primary: the 100 test 1.5 kg boxes
| run | threshold | read 1.5kg (K) | read 0.5kg (M) | missed | other |
|---|--:|--:|--:|--:|--:|
| Run 1 (reference) | 0.80 | 36 | 40 | 21 | 3 |
| 2C control | 0.67 | 40 | 51 | 6 | 3 |
| 2A colour | 0.53 | 72 | 25 | 0 | 3 |

Paired, 2A vs 2C: b = 26 misread by 2C only, c = 0 misread by 2A only; one-sided exact McNemar p = 0.0000.

| criterion | holds |
|---|---|
| void: M(2C) <= 20 | False |
| (a) M(2A) <= floor(M(2C) / 2) | True |
| (b) p < 0.05 | True |
| (c) K(2A) - K(2C) >= (M(2C) - M(2A)) / 2 | True |
| (d) G1, G2, G3 | True |

### Guardrails, 2A against 2C
| | 2C | 2A | margin | holds |
|---|--:|--:|--:|---|
| G1 test mAP@50-95 | 0.748 | 0.762 | -0.02 | True |
| G2 val mAP@50-95 (best) | 0.823 | 0.821 | -0.02 | True |
| G3 test AP 1kg | 0.690 | 0.686 | -0.05 | True |
| G3 test AP 2kg | 0.733 | 0.722 | -0.05 | True |
| G3 test AP 2.5kg | 0.688 | 0.692 | -0.05 | True |
| G3 test AP 5kg | 0.707 | 0.725 | -0.05 | True |
| G3 test AP 10kg | 0.813 | 0.829 | -0.05 | True |
| G3 test AP 15kg | 0.830 | 0.825 | -0.05 | True |
| G3 test AP 20kg | 0.842 | 0.832 | -0.05 | True |
| G3 test AP 25kg | 0.875 | 0.881 | -0.05 | True |
| G3 test AP zacisk | 0.820 | 0.811 | -0.05 | True |

### Secondary (decides nothing)
| run | M yellow | K yellow | M blue | K blue | 0.5kg precision | test mAP@50 | test mAP@50-95 | val mAP@50-95 | val to test gap |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Run 1 | 40 | 17 | 0 | 19 | 0.715 | 0.953 | 0.741 | 0.829 | 0.088 |
| 2C | 51 | 17 | 0 | 23 | 0.660 | 0.960 | 0.748 | 0.823 | 0.075 |
| 2A | 25 | 49 | 0 | 23 | 0.769 | 0.981 | 0.762 | 0.821 | 0.059 |

Full test confusion tables are in the JSON (`runs.<run>.confusion`).

### What this cannot show
- The test date is no longer untouched: its errors chose the question. These numbers say whether the known failure is fixed on this date, not how well the model does on a new one.
- One seed per arm. McNemar covers which boxes are in the test set, not training randomness; 2C against Run 1 is the only look at that.
- The augmentation simulates the colour of the light. A fix does not show that white balance or illumination was the real cause.
