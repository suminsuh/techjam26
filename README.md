# RIFT — Robust Image Forgery Tracer

TikTok TechJam 2026 · Track 5 · *Robust Detection of AI-Generated Images Under Real-World Transformations*

RIFT is a hackathon-scale AIGC detector built around the thing the brief actually asks for: **keep the decision stable after JPEG, blur, thumbnailing, noise, filters, and crops** — not just look good on clean lab images.

Most teams will fine-tune a CNN on CIFAKE, quote a clean accuracy, and watch it collapse after a WhatsApp re-encode. This repo starts from the opposite assumption.

## Why this design

Research on AIGC detection is consistent on one point: **forensic frequency cues are strong on pristine images and fragile after compression**. Semantic / spatial backbones degrade more slowly. A detector that only uses one of those families is either brittle or weak.

RIFT therefore does four things on purpose:

1. **Dual stream.** A spatial encoder looks at texture and structure. A forensic encoder looks at SRM residuals, log-FFT magnitude, and FFT **phase** (phase survives JPEG quantization better than magnitude).
2. **Trust gate.** A 2-way softmax decides how much to believe each stream. The demo prints this. Error analysis uses it. You can show a JPEG-30 image flipping the gate from forensic → spatial.
3. **Consistency training.** Each image is seen under two independent official transforms. The AIGC probability is pulled together. That is the training-time version of “this photo got reposted.”
4. **Fixed-threshold evaluation.** One cutoff is fit on **clean** validation (default: 5% FPR) and **frozen** for every transform. Retuning per JPEG quality inflates robustness and is not how a platform ships.

Parameter budget stays far under the **< 2B** rule: default `tiny` ≈ 0.4M, `efficientnet_b0` ≈ 5M, `convnext_tiny` ≈ 28M plus a small forensic head.

## What is in the box

| Path | Role |
|------|------|
| `src/rift/transforms.py` | Official TechJam transform grid, exact parameters |
| `src/rift/features.py` | SRM + FFT magnitude + phase front-end |
| `src/rift/models/dual_stream.py` | Dual-stream model + gated fusion |
| `src/rift/engine/train.py` | Official-aug + consistency training |
| `src/rift/engine/evaluate.py` | Frozen-threshold robustness table (deliverable 4) |
| `src/rift/engine/errors.py` | FP / FN note (deliverable 5) |
| `scripts/predict.py` | Required scorer: folder → `{image_path, pred}` JSON |
| `scripts/demo.py` | Gradio: one upload, full transform probe, Grad-CAM |

## Setup

Python 3.10–3.13, Windows / macOS / Linux.

```powershell
cd C:\Users\aiinapp\hackathons\techjam26
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[demo,dev]"
python scripts/prepare_data.py --samples
pytest -q
```

If `pip install -e .` is slow, `pip install -r requirements.txt` then run scripts with `PYTHONPATH=src` (already handled inside `scripts/`).

Untrained weights will warn and score ~0.5. That is expected. Do not submit those JSON files.

## Publishing to GitHub

Safe to commit: source, configs, tests, `data/samples`, README.  
Do **not** commit `.venv/`, `outputs/`, `checkpoints/`, downloaded datasets, or `.pt` weights.

```powershell
git add .
git status   # confirm no .venv, no checkpoints, no cifake
git commit -m "Add RIFT starter for TechJam Track 5 AIGC detection."
gh repo create rift --public --source=. --remote=origin --push
```

Or create an empty GitHub repo in the browser and `git remote add origin <url>` then `git push -u origin HEAD`.

## Data

Do **not** train on the organizers' WildFake demonstration holdout:

- Non-AIGC: COCO val2017 (4998)
- AIGC: DALL·E Advanced (8843)

That split is demo-only and will not count toward the score. See `data/README.md`.

Suggested path for the three days:

1. **Day 1** — CIFAKE to prove the pipeline (`tiny` backbone, CPU or one GPU).
2. **Day 2** — SID_Set or a WildFake *non-holdout* slice + `convnext_tiny`.
3. **Day 3** — robustness table, error note, Gradio video, Devpost.

```powershell
python scripts/train.py --config configs/default.yaml --train_dir data/cifake/train --val_dir data/cifake/test
python scripts/train.py --config configs/convnext_tiny.yaml --train_dir data/cifake/train --val_dir data/cifake/test
```

## Predict (required deliverable)

```powershell
python scripts/predict.py --input_dir path\to\images --output predictions.json
```

```json
[
  {"image_path": "path/to/img_001.jpg", "pred": 0.87},
  {"image_path": "path/to/img_002.jpg", "pred": 0.12}
]
```

`pred` is P(image is AIGC). Add `--aux` if you also want the gate weights.

## Robustness table + error analysis

```powershell
python scripts/evaluate.py --config configs/default.yaml --data_dir data/cifake/test --output_dir outputs
python scripts/error_analysis.py --metrics outputs/metrics.json --preds outputs/predictions_by_condition.json --output outputs/error_analysis.md
```

`outputs/robustness_table.md` is the compact clean-vs-transformed summary the brief asks for. Quote **accuracy / AUROC / FPR / FNR at the frozen threshold**, not a freshly tuned cutoff per row.

## Demo video loop

```powershell
python scripts/demo.py --config configs/default.yaml --checkpoint checkpoints/best.pt
```

Record: upload → score → Grad-CAM → the robustness dataframe on the same image. That is the end-to-end story.

## Team: what to do next (do these, not a rewrite)

The base is intentionally complete enough to train and incomplete enough to win on. Highest-leverage upgrades, in order:

1. **Swap the spatial backbone** to `convnext_tiny` once CIFAKE overfits. Config is already there.
2. **Optional frozen CLIP stream** (ViT-B/32 is ~151M, still legal). CLIP linear probes generalize to unseen generators; add it as a third gated expert, do not replace the forensic stream.
3. **Generator-ID auxiliary head** on SID_Set / WildFake. Forces features that still work when the test generator is new.
4. **Test-time augmentation** (`predict.tta`) averaging clean + JPEG-90 + mild crop. Cheap robustness at inference.
5. **Calibration** (temperature scaling on clean val) so `pred` is a real probability, not an overconfident logit.
6. **Hard-negative mining** from the error note: real screenshots, memes, and heavily compressed camera photos.

Do not: train a 1.5B vision model, build a production moderation platform, or chase CIFAKE test accuracy as the headline metric.

## Limitations (seed text for the README / Devpost)

- Untrained weights are random; run `train.py` before any demo.
- CIFAKE is 32×32 Stable Diffusion 1.4 vs CIFAR-10. It is a pipeline check, not a generalization claim.
- Forensic cues and JPEG are in tension. The gate is a mitigation, not a proof.
- We have not yet measured the official WildFake holdout. When we do, we will report the frozen threshold, not a retuned one.
- False positives on heavily compressed authentic images are the product risk we are optimizing against (`target_fpr: 0.05`).

## Team contributions

| Member | Area |
|--------|------|
| _name_ | Model / training |
| _name_ | Evaluation / robustness table |
| _name_ | Demo / video / write-up |

## License

MIT. Datasets remain under their own licenses — check CIFAKE, SID_Set, and WildFake before you publish weights trained on them.
