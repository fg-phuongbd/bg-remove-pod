import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def red_circle():
    im = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse((50, 50, 350, 350), fill=(220, 30, 30, 255))
    return im


@pytest.fixture
def red_square():
    im = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle((100, 100, 199, 199), fill=(220, 30, 30, 255))
    return im
