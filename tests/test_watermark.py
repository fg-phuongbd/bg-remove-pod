import numpy as np
from PIL import Image, ImageDraw

import pipeline


def _stamp(im: Image.Image, size: int, right: int, bottom: int) -> Image.Image:
    """Lay the Gemini sparkle over `im` the way Gemini does: white at the template's alpha."""
    t = np.asarray(pipeline._sparkle(size)).astype(np.float32)[:, :, None] / 255.0
    a = np.asarray(im.convert("RGB")).astype(np.float32).copy()
    y, x = im.height - bottom - size, im.width - right - size
    a[y:y + size, x:x + size] = a[y:y + size, x:x + size] * (1 - t) + 255.0 * t
    return Image.fromarray(a.round().astype(np.uint8), "RGB").convert("RGBA")


def _art(size: int, bg: tuple[int, int, int]) -> Image.Image:
    im = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(im)
    d.ellipse((size // 4, size // 4, size * 3 // 4, size * 3 // 4), fill=(200, 28, 50))
    return im.convert("RGBA")


def _diff(a: Image.Image, b: Image.Image) -> float:
    return float(np.abs(np.asarray(a.convert("RGB")).astype(int) - np.asarray(b.convert("RGB")).astype(int)).max())


def test_removes_the_sparkle_on_a_dark_background():
    clean = _art(2048, (36, 36, 36))
    out, found = pipeline.remove_watermark(_stamp(clean, 96, 192, 192))
    assert found == "Gemini"
    assert _diff(out, clean) <= 3


def test_removes_a_smaller_sparkle_on_a_light_background():
    clean = _art(1024, (232, 225, 205))
    out, found = pipeline.remove_watermark(_stamp(clean, 48, 32, 32))
    assert found == "Gemini"
    assert _diff(out, clean) <= 3


def test_restores_design_under_the_sparkle():
    clean = _art(2048, (36, 36, 36))
    ImageDraw.Draw(clean).rectangle((1700, 1700, 2047, 2047), fill=(200, 28, 50, 255))  # ink under it
    out, found = pipeline.remove_watermark(_stamp(clean, 96, 192, 192))
    assert found == "Gemini"
    assert _diff(out, clean) <= 3


def test_leaves_a_picture_without_the_sparkle_alone():
    clean = _art(2048, (36, 36, 36))
    ImageDraw.Draw(clean).ellipse((1760, 1760, 1856, 1856), fill=(255, 255, 255, 255))  # a white dot is design
    out, found = pipeline.remove_watermark(clean)
    assert found is None
    assert out is clean
