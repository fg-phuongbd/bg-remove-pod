import pytest
from PIL import Image

import pipeline


def test_target_box_px():
    assert pipeline.target_box_px("30x40") == (3543, 4724)
    assert pipeline.target_box_px("21x29.7") == (2480, 3508)
    assert pipeline.target_box_px("4500x5100") == (4500, 5100)
    assert pipeline.target_box_px("4.500x5.100") == (4500, 5100)


def test_place_on_canvas(red_circle):
    out = pipeline.place_on_canvas(red_circle.resize((100, 200)), (300, 400))
    assert out.size == (300, 400)
    assert out.getpixel((150, 200))[3] == 255 and out.getpixel((10, 10))[3] == 0


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


def test_fill_holes_only_enclosed(red_circle):
    from PIL import ImageDraw

    cut = red_circle.copy()
    ImageDraw.Draw(cut).ellipse((180, 180, 220, 220), fill=(0, 0, 0, 0))  # hole inside the disc
    original = Image.new("RGBA", (400, 400), (255, 255, 255, 255))
    out = pipeline.fill_holes(cut, original)
    assert out.getpixel((200, 200)) == (255, 255, 255, 255)  # hole restored from original
    assert out.getpixel((5, 5))[3] == 0  # outside background stays transparent
