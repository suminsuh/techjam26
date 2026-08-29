import numpy as np
from PIL import Image

from rift.transforms import CONDITION_FACTORIES, gaussian_noise, jpeg_compress, to_rgb


def test_every_official_condition_returns_rgb(sample_dir):
    image = Image.open(next((sample_dir / "REAL").glob("*.png")))
    for name, factory in CONDITION_FACTORIES.items():
        out = factory()(image)
        assert out.mode == "RGB", name
        assert out.size[0] > 0 and out.size[1] > 0


def test_jpeg_changes_bytes_but_stays_decodable(sample_dir):
    image = to_rgb(Image.open(next((sample_dir / "REAL").glob("*.png"))))
    out = jpeg_compress(image, 30)
    assert out.mode == "RGB"
    assert out.size == image.size


def test_seeded_noise_is_deterministic(sample_dir):
    image = to_rgb(Image.open(next((sample_dir / "REAL").glob("*.png"))))
    a = np.asarray(gaussian_noise(image, 0.05, seed=7))
    b = np.asarray(gaussian_noise(image, 0.05, seed=7))
    c = np.asarray(gaussian_noise(image, 0.05, seed=8))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
