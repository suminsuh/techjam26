# RIFT: Robust Image Forgery Tracer

TikTok TechJam 2026, Track 5: robust detection of AI-generated images after JPEG, blur, resize, noise, colour jitter, and crop.

RIFT scores **P(AI-generated)** for a folder of images and keeps **one** decision cutoff after those re-share transforms. It does not retune the threshold per JPEG quality.

## What we submitted

Frozen **CLIP ViT-B/32** (spatial) plus a small CNN on **SRM residuals, log-FFT magnitude, and FFT phase** (forensic). A softmax **trust gate** mixes the two. CLIP stays frozen. About **88.3M** parameters total, **~850k** trainable, under the 2B limit.

Trained on **SID_Set** real vs full-synthetic only (tampered splices dropped): **20,000** images (10k / 10k), 3 epochs, official-transform augmentations, two-view consistency. Config: `configs/clip_sidset.yaml`. Checkpoint (not in git): `checkpoints/sidset/best_20k.pt`.

CIFAKE was used only to debug the pipeline. The official WildFake holdout (COCO val2017 + DALL-E Advanced) is **eval-only** and was **not** used for training. We have not reported a holdout number.

On a balanced **2,000**-image SID_Set val slice, threshold **0.0317** (5% FPR on that same clean split), frozen for every row:

| Condition | Acc | AUROC | FPR | FNR |
|---|---:|---:|---:|---:|
| Clean | 97.6% | 0.9996 | 4.7% | 0.1% |
| JPEG 30 | 97.4% | 0.9993 | 4.9% | 0.3% |
| Blur σ=2 | 97.3% | 0.9993 | 5.2% | 0.3% |
| Resize 0.25× | 97.7% | 0.9992 | 4.1% | 0.5% |
| Noise σ=0.10 | 95.9% | 0.9977 | 7.5% | 0.8% |

That table is **in-domain SID_Set**, not unseen generators. On this checkpoint the gate is almost entirely CLIP (spatial ~1.00). Treat 97.6% as ranking quality on this split, not as a Track 5 holdout score. Full grid: JPEG 90/70/50/30, blur 0.5/1.0/2.0, resize 0.5/0.25, noise 0.02/0.05/0.10, colour jitter, centre crop 80%.

## Setup

Python 3.10–3.13. GPU recommended (trained on an RTX 4070 8GB laptop).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"
$env:PYTHONIOENCODING = "utf-8"
python scripts/prepare_data.py --samples
python -m pytest -q
```

On Windows, SID_Set downloads are more reliable with `$env:HF_HUB_DISABLE_XET = "1"`.

Weights are not in this repo. Place `best_20k.pt` (or `best.pt`) under `checkpoints/sidset/` after training, or copy the file your team already has.

## Required scorer

```powershell
python scripts/predict.py --config configs/clip_sidset.yaml --checkpoint checkpoints/sidset/best_20k.pt --input_dir path\to\images --output predictions.json
```

```json
[
  {"image_path": "path/to/img_001.jpg", "pred": 0.87},
  {"image_path": "path/to/img_002.jpg", "pred": 0.12}
]
```

`pred` is P(the image is AI-generated). `--aux` also writes gate weights.

## Demo

```powershell
python scripts/demo.py --config configs/clip_sidset.yaml --checkpoint checkpoints/sidset/best_20k.pt
```

Upload one image. The page shows the call (likely AI-generated vs likely authentic), the frozen cutoff, and the same image after official transforms.

## Train / eval

```powershell
python scripts/cache_sidset.py
python scripts/train.py --config configs/clip_sidset.yaml
python scripts/evaluate.py --config configs/clip_sidset.yaml --checkpoint checkpoints/sidset/best_20k.pt --output_dir outputs/sidset
```

Holdout eval (needs `techjam-aigc` Hugging Face access). **Do not fit a new threshold:**

```powershell
python scripts/eval_holdout.py
```

`configs/default.yaml` is a tiny-CNN smoke test on ImageFolder data. `configs/clip_cifake.yaml` is the CIFAKE probe. Neither is the submitted detector.

## Layout

| Path | Role |
|------|------|
| `src/rift/transforms.py` | Official transform grid |
| `src/rift/features.py` | SRM + FFT magnitude + phase |
| `src/rift/models/dual_stream.py` | Dual-stream model + gate |
| `src/rift/models/clip_stream.py` | Frozen CLIP encoder |
| `src/rift/engine/train.py` | Official augs + consistency |
| `src/rift/engine/evaluate.py` | Frozen-threshold robustness table |
| `src/rift/engine/errors.py` | FP / FN note |
| `scripts/predict.py` | Folder to `{image_path, pred}` JSON |
| `scripts/demo.py` | Gradio demo |
| `configs/clip_sidset.yaml` | Submitted train/eval config |

## Limitations

- Untrained or missing checkpoints score around 0.5. Pass `--checkpoint`.
- SID_Set numbers are real vs full-synthetic from the same corpus as training. They are not official-holdout accuracy.
- The trained gate puts almost all weight on CLIP. The forensic stream is in the graph; this run barely uses it.
- False positives on compressed or unusual authentic photos are the product risk behind the 5% FPR cutoff.
- Official holdout must not be used as `train_dir`. See `data/README.md`.

## License

MIT for this code. CIFAKE, SID_Set, WildFake, COCO, and CLIP weights keep their own terms. Do not publish those datasets in git.
