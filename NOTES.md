# Build Notes — Custom Object Detector

Working notes: what broke, what you tried, why you chose X over Y.
Not for recruiters — for you, six months from now, in an interview.

Keep it rough. Rough is the point.

---

## Settled design (2026-09-23)

| Decision | Chose | Why |
|---|---|---|
| Domain | Gym weight plates | Every image is my own → publishable dataset, clean-clone reproducible. Not a COCO class. |
| Detector classes | `25kg` `20kg` `15kg` `10kg` `5kg` `2.5kg` | Six visually meaningful classes; fits a 2-session budget. |
| Loaded weight | **Derived task, not a detection class** | Detect plates → class→kg → group by sleeve → sum → + bar. Keeps detection and task correctness as separate layers. |
| Detector | RF-DETR-S (`rfdetr`, Apache-2.0) | Repo stays MIT. Benchmarked on RF100-VL, i.e. small custom datasets. 32M params → trains on a Kaggle T4, infers on CPU. M is the step up if S underfits. |
| Ground truth for load | Recorded at capture, from what is physically on the bar | If the "true total" came from the box labels, the task metric would just re-score the detector against its own labels. |
| Dataset | Gym A → train/val/test; Gym B → test only | Gym B measures how far the model degrades on plates it has never seen. |
| Camera rule | ~30–45° to the sleeve | End-on shows only the outer plate; side-on shows edges, and iron plates look identical edge-on. |
| Occlusion | Photos with any fully hidden plate → separate *occluded* group | Reported on its own, not silently counted as failures. |
| Left/right rule | Symmetric loading assumed. One sleeve visible → per-side sum × 2. Both visible → sum each, flag a mismatch. | Standard gym practice; stated so the metric is unambiguous. |
| Split | By `load_id`, never by photo; split fixed in `data/CAPTURE_PROTOCOL.md` before training | Six photos of one load are near-duplicates — a random split leaks. |

**Metrics, two layers that stay separate:**
1. Detection — mAP@50, mAP@50–95, precision, recall (per class).
2. Task correctness — % of images whose estimated total exactly matches the recorded total,
   plus mean absolute error in kg. An independent downstream test, not another box score.

## Licence check (verified upstream 2026-09-23)

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

---

## Rejected approaches

| Approach | Why rejected |
|---|---|
| Document marks (signature/stamp/QR) | Strongest capstone link, but my real documents carry personal data and can't be published — breaks clean-clone reproducibility. |
| Candlestick chart patterns | Boxes are subjective → mAP measures labeller consistency, not the model. Pretrained models already exist. |
| Indian currency notes | Common Kaggle project; mAP likely saturates near 0.95 and proves little. |
| FMCG shelf SKUs, PCB parts, road hazards | Fine-grained classes, tiny objects or heavy occlusion — too risky for ~200 images in 2 sessions. |
| Bar load as a detection class | Mixes detection with arithmetic; loses the independent task metric. |
| Ultralytics YOLO | AGPL-3.0 (see licence check). |
| Roboflow's random train/val/test split | Leaks near-duplicate photos of one load across splits. |

## Open questions

- [ ] Is a second gym (Gym B) actually reachable? If not, the single-gym limitation goes in
      the README plainly.
- [ ] Does Gym A have 4 of every denomination? The full schedule needs it.
- [ ] Add a stricter third task metric — exact plate-set match per sleeve — since totals
      collide? Decide before training, not after seeing results.
