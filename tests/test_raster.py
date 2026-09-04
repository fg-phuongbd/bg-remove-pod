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
