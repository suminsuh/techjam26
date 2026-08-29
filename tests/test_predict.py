import json
from pathlib import Path

from rift.config import load_config
from rift.engine.predict import predict_directory
from rift.metrics import choose_threshold, evaluate_scores
import numpy as np


def test_predict_json_contract(sample_dir, tmp_path):
    cfg = load_config()
    cfg["model"] = {"spatial_backbone": "tiny", "pretrained": False, "embed_dim": 64, "dropout": 0.0}
    cfg["predict"] = {"batch_size": 2, "tta": False, "checkpoint": ""}
    out = tmp_path / "predictions.json"
    records = predict_directory(sample_dir, cfg, checkpoint=None, output_path=out)
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert len(loaded) == 4
    for row in loaded:
        assert set(row) == {"image_path", "pred"}
        assert Path(row["image_path"]).exists()
        assert 0.0 <= row["pred"] <= 1.0
    assert len(records) == 4
    assert all("/" in row["image_path"] or row["image_path"].count("\\") == 0 for row in loaded)


def test_fixed_threshold_does_not_retune():
    y = np.array([0, 0, 1, 1])
    clean = np.array([0.1, 0.2, 0.8, 0.9])
    jpeg = np.array([0.2, 0.25, 0.55, 0.6])
    t = choose_threshold(y, clean, target_fpr=0.5)
    clean_rep = evaluate_scores(y, clean, t)
    jpeg_rep = evaluate_scores(y, jpeg, t)
    assert clean_rep.threshold == jpeg_rep.threshold
    assert np.isfinite(t)


def test_threshold_is_finite_on_tied_scores():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.51, 0.51, 0.51, 0.51])
    t = choose_threshold(y, scores, target_fpr=0.05)
    assert np.isfinite(t)
