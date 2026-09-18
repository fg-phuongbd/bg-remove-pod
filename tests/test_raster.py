from pathlib import Path

from PIL import Image

import pipeline


def test_upscale_fallback(monkeypatch):
    monkeypatch.setattr(pipeline, "REALESRGAN_BIN", Path("/nonexistent/realesrgan"))
    out = pipeline.upscale(Image.new("RGBA", (100, 50), (255, 0, 0, 255)))
    assert out.size == (400, 200)


def test_flatten_raster(red_circle):
    px = red_circle.load()
    for x in range(60, 340, 3):
        px[x, 200] = (230, 40, 40, 255)
    out = pipeline.flatten_raster(red_circle, colors=2)
    assert len({p[:3] for p in out.get_flattened_data() if p[3] == 255}) <= 2


def test_tighten_alpha():
    im = Image.new("RGBA", (3, 1))
    im.putpixel((0, 0), (0, 0, 0, 90)); im.putpixel((1, 0), (0, 0, 0, 128)); im.putpixel((2, 0), (0, 0, 0, 170))
    out = pipeline.tighten_alpha(im)
    assert out.getpixel((0, 0))[3] == 0 and out.getpixel((2, 0))[3] == 255
    assert 100 < out.getpixel((1, 0))[3] < 160


def _halftone(size=240, spacing=6, r=2):
    """Chấm bi trắng đều trên nền trong suốt: thứ Real-ESRGAN biến thành vệt lông xù."""
    from PIL import Image, ImageDraw
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for y in range(spacing, size - spacing, spacing):
        for x in range(spacing, size - spacing, spacing):
            d.ellipse((x - r, y - r, x + r, y + r), fill=(240, 240, 240, 255))
    return im


def test_is_grainy_tells_halftone_from_flat_and_photo(red_circle):
    import numpy as np
    from PIL import Image
    import pipeline

    assert pipeline.is_grainy(_halftone()) is True
    assert pipeline.is_grainy(red_circle) is False
    rng = np.random.default_rng(0)                                   # ảnh chụp: gradient + nhiễu nhẹ
    a = np.dstack([np.tile(np.linspace(40, 220, 240), (240, 1))] * 3 + [np.full((240, 240), 255)])
    a[:, :, :3] += rng.normal(0, 3, (240, 240, 3))
    assert pipeline.is_grainy(Image.fromarray(a.clip(0, 255).astype(np.uint8), "RGBA")) is False


def test_detect_style_names_grain_before_flat_or_detail():
    import pipeline

    assert pipeline.detect_style(_halftone()) == "grain"


def test_lanczos_upscale_never_calls_the_binary(monkeypatch, red_circle):
    import subprocess
    import pipeline

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("gọi binary")))
    out = pipeline.upscale(red_circle, model="lanczos")
    assert out.size == (1600, 1600)
