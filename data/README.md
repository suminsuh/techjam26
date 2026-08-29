# Data

Keep large datasets out of git. Point `configs/default.yaml` at local folders.

## Folder convention

```
data/cifake/train/REAL/*.jpg
data/cifake/train/FAKE/*.jpg
data/cifake/test/REAL/*.jpg
data/cifake/test/FAKE/*.jpg
```

Accepted class folder names: `REAL`/`real`/`authentic` and `FAKE`/`fake`/`aigc`/`synthetic`/`generated`.

## Holdout (do not train)

The organizers' demonstration subset of WildFake is **off-limits for training**:

| Split    | Source            | Count |
|----------|-------------------|------:|
| Non-AIGC | COCO val2017      |  4998 |
| AIGC     | DALL·E Advanced   |  8843 |

Use it only when you need a public demo number. Put it under `data/wildfake_holdout/` if you download it, and never list that path as `train_dir`.

## Bootstrap placeholders

```
python scripts/prepare_data.py --samples
```

Writes four tiny PNGs under `data/samples/` so `predict.py` and pytest work before any dataset download.
