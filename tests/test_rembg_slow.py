import pytest
from PIL import Image, ImageDraw

import pipeline


@pytest.mark.slow
def test_remove_bg_real_model():
    im = Image.new("RGB", (512, 512), (255, 255, 255))
    ImageDraw.Draw(im).ellipse((100, 100, 412, 412), fill=(220, 30, 30))
    out = pipeline.remove_bg(im)
    assert out.mode == "RGBA"
    assert out.getpixel((5, 5))[3] < 20
    assert out.getpixel((256, 256))[3] > 235
