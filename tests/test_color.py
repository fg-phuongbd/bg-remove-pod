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


def test_small_highlight_survives_soft_edges():
    """A tiny white dot with blurred edges inside a big dark disc must stay white."""
    from PIL import ImageDraw, ImageFilter

    im = Image.new("RGB", (400, 400), (255, 170, 60))
    d = ImageDraw.Draw(im)
    d.ellipse((100, 100, 300, 300), fill=(40, 40, 60))
    d.ellipse((150, 150, 190, 190), fill=(255, 255, 255))
    im = im.filter(ImageFilter.GaussianBlur(1.2))
    im.putalpha(255)
    out = pipeline.quantize(im, 6, binary_alpha=True)
    r, g, b, _ = out.getpixel((170, 170))
    assert min(r, g, b) > 230, (r, g, b)
    assert out.getpixel((200, 250))[:3] == out.getpixel((120, 200))[:3]  # disc stays one flat color


def test_noise_shades_are_merged_even_under_budget():
    """Two oranges only a few Lab units apart collapse into one, even with a generous budget."""
    im = Image.new("RGBA", (200, 100), (251, 168, 60, 255))
    for x in range(100, 200):
        for y in range(100):
            im.putpixel((x, y), (253, 155, 66, 255))
    out = pipeline.quantize(im, 12, binary_alpha=True)
    assert len(_opaque_colors(out)) == 1


def test_scattered_noise_shade_absorbed_but_compact_and_outline_kept():
    """Teal specks on a blue body (ΔE≈10) vanish; a small red mouth and a thin black outline survive."""
    import numpy as np
    from PIL import ImageDraw

    rng = np.random.default_rng(1)
    im = Image.new("RGBA", (600, 600), (60, 129, 199, 255))
    d = ImageDraw.Draw(im)
    for _ in range(400):  # scattered 4-7 px specks of a nearby shade
        x, y, s = rng.integers(0, 590), rng.integers(0, 590), rng.integers(4, 8)
        d.ellipse((x, y, x + s, y + s), fill=(59, 146, 202, 255))
    d.ellipse((250, 400, 350, 440), fill=(200, 62, 82, 255))  # compact small red
    d.rectangle((50, 50, 550, 300), outline=(20, 20, 20, 255), width=3)  # thin outline
    out = pipeline.quantize(im, 12, binary_alpha=True)
    cols = _opaque_colors(out)
    assert not any(abs(c[1] - 146) < 6 and abs(c[2] - 202) < 6 for c in cols), cols  # teal gone
    assert out.getpixel((300, 420))[:3] == out.getpixel((300, 420))[:3] and out.getpixel((300, 420))[0] > 150  # red kept
    assert out.getpixel((51, 175))[0] < 60  # outline kept
    assert len(cols) == 3, cols


def test_fringe_takes_neighbour_color():
    """A light, semi-transparent fringe around a dark shape becomes the shape's color."""
    import numpy as np

    a = np.zeros((60, 60, 4), dtype=np.uint8)
    a[10:50, 10:50] = (40, 40, 60, 255)
    a[8:52, 8:52][a[8:52, 8:52, 3] == 0] = (230, 230, 230, 120)  # 2 px light fringe
    out = pipeline.quantize(Image.fromarray(a, "RGBA"), 6, binary_alpha=False)
    assert out.getpixel((9, 30))[:3] == out.getpixel((30, 30))[:3]
    assert 0 < out.getpixel((9, 30))[3] < 255
    assert len(_opaque_colors(out)) == 1
