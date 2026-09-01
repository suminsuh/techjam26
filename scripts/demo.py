"""Local Gradio demo: score one image and probe official transforms.

This is the visual you want in the video: the *same* upload, re-encoded /
blurred / cropped, with a frozen decision and a trust-gate explanation.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from rift.config import load_config  # noqa: E402
from rift.engine.explain import gate_story, gradcam, overlay_heatmap  # noqa: E402
from rift.engine.predict import load_checkpoint  # noqa: E402
from rift.preprocess import pil_to_tensor  # noqa: E402
from rift.seed import resolve_device  # noqa: E402
from rift.transforms import CONDITION_FACTORIES, to_rgb  # noqa: E402


def _tensorize(image: Image.Image, size: int) -> torch.Tensor:
    return pil_to_tensor(image, size).unsqueeze(0)


def _load_threshold(default: float = 0.5) -> float:
    import json

    for metrics in (
        ROOT / "outputs" / "sidset" / "metrics.json",
        ROOT / "outputs" / "metrics.json",
    ):
        if not metrics.exists():
            continue
        try:
            return float(json.loads(metrics.read_text(encoding="utf-8"))["threshold"])
        except (KeyError, TypeError, ValueError):
            continue
    return default


def _verdict(pred: float, threshold: float) -> str:
    return "Likely AI-generated" if pred >= threshold else "Likely authentic"


TRANSFORM_LABELS = {
    "clean": "Original",
    "jpeg_70": "JPEG 70",
    "jpeg_30": "JPEG 30",
    "blur_1.0": "Blur",
    "resize_0.25": "Thumbnail",
    "noise_0.05": "Noise",
    "color_jitter": "Color shift",
    "center_crop_0.8": "Center crop",
}


def run_demo(config: str, checkpoint: str | None, share: bool) -> None:
    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit("Gradio is missing. Run: pip install gradio") from exc

    cfg = load_config(config)
    device = resolve_device(cfg.get("device", "auto"))
    size = int(cfg.get("image_size", 224))
    threshold = _load_threshold()
    model = load_checkpoint(checkpoint or cfg.get("predict", {}).get("checkpoint"), cfg, device)

    @torch.no_grad()
    def _score(pil: Image.Image):
        tensor = _tensorize(pil, size).to(device)
        logit, aux = model(tensor, return_aux=True)
        pred = float(torch.sigmoid(logit).item())
        gate = aux["gate"][0]
        return pred, gate

    def analyze(image: np.ndarray, show_cam: bool):
        if image is None:
            return "Upload an image first.", None, []
        pil = to_rgb(Image.fromarray(image.astype(np.uint8)))
        pred, gate = _score(pil)
        label = _verdict(pred, threshold)
        story = (
            f"## **{label}**\n\n"
            f"Score **{pred:.3f}** (0 = authentic, 1 = AI-generated). "
            f"Cutoff **{threshold:.3f}**, kept the same for every transform.\n\n"
            f"{gate_story(gate)}"
        )
        cam_img = None
        if show_cam:
            try:
                heatmap = gradcam(model, _tensorize(pil, size).to(device))
                cam_img = overlay_heatmap(pil.resize((size, size)), heatmap)
            except Exception:
                cam_img = None

        rows = []
        for name in [
            "clean",
            "jpeg_70",
            "jpeg_30",
            "blur_1.0",
            "resize_0.25",
            "noise_0.05",
            "color_jitter",
            "center_crop_0.8",
        ]:
            transformed = CONDITION_FACTORIES[name]()(pil)
            t_pred, t_gate = _score(transformed)
            rows.append(
                [
                    TRANSFORM_LABELS.get(name, name),
                    round(t_pred, 3),
                    _verdict(t_pred, threshold),
                    round(float(t_gate[0]), 2),
                    round(float(t_gate[1]), 2),
                ]
            )
        return story, cam_img, rows

    table_headers = ["transform", "score", "verdict", "appearance", "traces"]
    demo = gr.Interface(
        fn=analyze,
        inputs=[
            gr.Image(type="numpy", label="Upload an image"),
            gr.Checkbox(label="Show where the model looked", value=True),
        ],
        outputs=[
            gr.Markdown(label="Result"),
            gr.Image(type="pil", label="Where the model looked"),
            gr.Dataframe(headers=table_headers, label="Same image after JPEG, blur, crop, and noise"),
        ],
        title="RIFT: is this photo AI-generated?",
        description=(
            "Upload a picture. RIFT scores it, then applies the official "
            "re-share transforms (JPEG, blur, thumbnail, noise, crop) with "
            "the same cutoff. Train a checkpoint first or the scores will "
            "look random."
        ),
    )
    demo.launch(share=share)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    run_demo(args.config, args.checkpoint, args.share)


if __name__ == "__main__":
    main()
