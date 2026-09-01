# Data

Keep large datasets out of git. Point configs at local folders.

## Submitted training set: SID_Set

`configs/clip_sidset.yaml` caches **real vs full_synthetic** only (label 2 / tampered is dropped).

```
data/sidset/train/{REAL,FAKE}/*.jpg
data/sidset/val/{REAL,FAKE}/*.jpg
```

```powershell
$env:HF_HUB_DISABLE_XET = "1"
python scripts/cache_sidset.py
```

The reported checkpoint used 10k images per class (20k train). The yaml can cache more (40k / 5k) if you have disk.

## CIFAKE (pipeline check only)

Hugging Face: `dragonintelligence/CIFAKE-image-dataset`. Labels on that mirror are 0=FAKE, 1=REAL; RIFT maps them so `pred` is still P(AI-generated). Prefer `configs/clip_cifake.yaml` over dumping 32x32 files unless you need ImageFolder.

```
data/cifake/train/{REAL,FAKE}/*.jpg
data/cifake/test/{REAL,FAKE}/*.jpg
```

Accepted class folder names: `REAL` / `real` / `authentic` / `coco` and `FAKE` / `fake` / `aigc` / `synthetic` / `generated`.

## Holdout (do not train)

Official WildFake demonstration subset:

| Split | Source | Count |
|----------|-------------------|------:|
| Authentic | COCO val2017 | 4998 |
| AI-generated | DALL-E Advanced | 8843 |

Eval only. Never set this as `train_dir`. Use `configs/holdout.yaml` and the frozen SID_Set threshold. The Hugging Face pack `techjam-aigc/wildfake-eval-subset` is org-private.

## Samples

```
python scripts/prepare_data.py --samples
```

Four tiny PNGs under `data/samples/` so pytest and `predict.py` run before any dataset download. They are not a training set.
