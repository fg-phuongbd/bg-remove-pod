import pytest
from PIL import Image

import pipeline


def test_target_box_px():
    assert pipeline.target_box_px("30x40") == (3543, 4724)
    assert pipeline.target_box_px("21x29.7") == (2480, 3508)


def test_fit_box():
    assert pipeline.fit_box(1000, 500, (3543, 4724)) == (3543, 1771)
    assert pipeline.fit_box(500, 1000, (3543, 4724)) == (2362, 4724)


def test_crop_to_content(red_square):
    out = pipeline.crop_to_content(red_square, margin=0.02)
    assert out.size == (104, 104)
    assert out.getpixel((52, 52)) == (220, 30, 30, 255)
    assert out.getpixel((0, 0))[3] == 0


def test_crop_empty_raises():
    with pytest.raises(pipeline.EmptyResult):
        pipeline.crop_to_content(Image.new("RGBA", (50, 50), (0, 0, 0, 0)))
