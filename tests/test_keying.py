import numpy as np
from PIL import Image, ImageDraw

import pipeline


def _dark_art():
    rng = np.random.default_rng(3)
    a = rng.normal(22, 4, (300, 300, 3)).clip(0, 60).astype(np.uint8)  # noisy near-black bg
    im = Image.fromarray(a, "RGB")
    d = ImageDraw.Draw(im)
    d.rectangle((50, 50, 150, 150), fill=(255, 255, 255))   # white
    d.rectangle((160, 50, 260, 150), fill=(128, 128, 128))  # mid grey
    d.rectangle((50, 160, 150, 260), fill=(0, 120, 200))    # blue
    return im


def test_detect_bg():
    assert pipeline.detect_bg(_dark_art()) == "black"
    assert pipeline.detect_bg(Image.new("RGB", (100, 100), (250, 250, 248))) == "ai"  # white is cut out, not keyed
    assert pipeline.detect_bg(Image.new("RGB", (100, 100), (120, 200, 90))) == "ai"
    assert pipeline.detect_bg(Image.new("RGBA", (100, 100), (0, 0, 0, 0))) == "none"


def test_key_black_reproduces_original_on_black():
    im = _dark_art()
    out = pipeline.key_bg(im, "black")
    assert out.getpixel((10, 10))[3] == 0                    # background gone
    assert out.getpixel((100, 100)) == (255, 255, 255, 255)  # white stays
    r, g, b, a = out.getpixel((210, 100))                    # grey -> white at ~half alpha
    assert (r, g, b) == (255, 255, 255) and 100 < a < 150
    r, g, b, a = out.getpixel((100, 210))                    # blue: un-premultiplied, max channel 255
    assert b == 255 and a > 180
    # compositing the result on black gives back the original color (within rounding + lo offset)
    comp = np.asarray(out).astype(float)
    back = comp[:, :, :3] * comp[:, :, 3:] / 255.0
    assert np.abs(back[210, 100] - [0, 120, 200]).max() < 16  # numpy is [y, x]


def test_key_white_mirror():
    im = Image.new("RGB", (100, 100), (255, 255, 255))
    ImageDraw.Draw(im).rectangle((20, 20, 80, 80), fill=(200, 30, 30))
    out = pipeline.key_bg(im, "white")
    assert out.getpixel((5, 5))[3] == 0
    r, g, b, a = out.getpixel((50, 50))
    assert a > 200 and r > 150 and g < 60


def test_detect_style(red_circle):
    assert pipeline.detect_style(red_circle) == "flat"
    assert pipeline.detect_style(pipeline.key_bg(_dark_art(), "black")) == "flat"  # solid blocks
    rng = np.random.default_rng(5)
    noisy = Image.fromarray(rng.integers(0, 255, (400, 400, 3), dtype=np.uint8), "RGB")
    assert pipeline.detect_style(noisy) == "detail"
