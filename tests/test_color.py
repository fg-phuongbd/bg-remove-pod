from PIL import Image

import pipeline


def _shaded():
    im = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    px = im.load()
    for y in range(10, 90):
        for x in range(10, 90):
            px[x, y] = (200 + (x % 50), 20, 20, 255)
    px[5, 5] = (10, 10, 10, 10)
    px[6, 6] = (10, 10, 10, 250)
    px[7, 7] = (10, 10, 10, 128)
    return im


def _opaque_colors(im):
    return {p[:3] for p in im.get_flattened_data() if p[3] == 255}


def test_quantize_binary_alpha():
    out = pipeline.quantize(_shaded(), 2, binary_alpha=True)
    assert len(_opaque_colors(out)) <= 2
    assert {p[3] for p in out.get_flattened_data()} <= {0, 255}


def test_quantize_soft_alpha():
    out = pipeline.quantize(_shaded(), 2, binary_alpha=False)
    assert out.getpixel((5, 5))[3] == 0
    assert out.getpixel((6, 6))[3] == 255
    assert out.getpixel((7, 7))[3] == 128
