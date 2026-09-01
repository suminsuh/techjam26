import torch

from rift.features import ForensicFrontend
from rift.models.dual_stream import DualStreamDetector
from rift.preprocess import IMAGENET_MEAN, IMAGENET_STD


def test_frontend_shape():
    x = torch.rand(2, 3, 64, 64)
    y = ForensicFrontend()(x)
    assert y.shape == (2, 9, 64, 64)
    assert torch.isfinite(y).all()


def test_tiny_model_forward_and_budget():
    model = DualStreamDetector(spatial_backbone="tiny", embed_dim=64, pretrained=False)
    x = torch.rand(2, 3, 64, 64)
    logit, aux = model(x, return_aux=True)
    assert logit.shape == (2,)
    assert aux["gate"].shape == (2, 2)
    assert torch.allclose(aux["gate"].sum(dim=1), torch.ones(2), atol=1e-5)
    assert model.param_count() < 2_000_000_000
    assert model.param_count() < 2_000_000
    assert model.normalize_spatial is False


def test_pretrained_style_backbone_normalizes_spatial_only():
    model = DualStreamDetector(spatial_backbone="tiny", embed_dim=32, pretrained=False)
    model.normalize_spatial = True
    captured: list[torch.Tensor] = []

    def hook(_mod, args):
        captured.append(args[0].detach().clone())

    handle = model.frontend.register_forward_pre_hook(lambda m, args: hook(m, args))
    x = torch.rand(1, 3, 64, 64)
    model(x)
    handle.remove()
    assert torch.allclose(captured[0], x)
    assert abs(float(model.pixel_mean[0, 0, 0, 0]) - IMAGENET_MEAN[0]) < 1e-6
    assert abs(float(model.pixel_std[0, 0, 0, 0]) - IMAGENET_STD[0]) < 1e-6


def test_gradcam_tiny_returns_map():
    from rift.engine.explain import gradcam

    model = DualStreamDetector(spatial_backbone="tiny", embed_dim=32, pretrained=False)
    model.eval()
    heat = gradcam(model, torch.rand(1, 3, 64, 64))
    assert heat.shape == (64, 64)
    assert heat.min() >= 0.0
    assert heat.max() <= 1.0 + 1e-6
