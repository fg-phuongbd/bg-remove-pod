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


def test_detect_bg_kinds():
    assert pipeline.detect_bg(_dark_art()) == "black"
    assert pipeline.detect_bg(Image.new("RGB", (100, 100), (250, 250, 248))) == "white"
    assert pipeline.detect_bg(Image.new("RGB", (100, 100), (120, 200, 90))) == "color"
    assert pipeline.detect_bg(Image.new("RGB", (100, 100), (253, 190, 213))) == "color"  # light pink is not white
    assert pipeline.detect_bg(Image.new("RGBA", (100, 100), (0, 0, 0, 0))) == "none"


def test_resolve_bg_by_shirt():
    r = pipeline.resolve_bg
    # (kind, shirt) -> (mode, refine)
    assert r("none", "auto") == ("none", False)
    assert r("black", "auto") == ("black", False)      # black art is for dark shirts by default
    assert r("black", "same") == ("black", False)
    assert r("black", "other") == ("ai", False)
    assert r("white", "auto") == ("ai", True)          # white art is cut out by default, edges refined
    assert r("white", "other") == ("ai", True)
    assert r("white", "same") == ("white", False)
    assert r("color", "auto") == ("ai", True)          # colored bg, other shirt: cutout + refine
    assert r("color", "other") == ("ai", True)
    assert r("color", "same") == ("color", False)


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


def test_key_color_removes_solid_colored_background():
    rng = np.random.default_rng(7)
    bg = np.array([253, 190, 213])
    a = (bg + rng.normal(0, 0.8, (300, 300, 3))).round().clip(0, 255).astype(np.uint8)  # light pink, AI noise
    im = Image.fromarray(a, "RGB")
    d = ImageDraw.Draw(im)
    d.rectangle((50, 50, 150, 150), fill=(255, 20, 147))    # hot pink, far from bg
    d.rectangle((160, 50, 260, 150), fill=(0, 0, 0))        # black
    d.rectangle((50, 160, 150, 260), fill=(253, 160, 200))  # slightly darker pink: partial alpha
    out = pipeline.key_bg(im, "color")
    assert out.getpixel((10, 10))[3] == 0                    # background gone
    assert out.getpixel((100, 100)) == (255, 20, 147, 255)   # design colors untouched
    assert out.getpixel((210, 100)) == (0, 0, 0, 255)        # black is design here, not shirt
    r, g, b, alpha = out.getpixel((100, 210))
    assert 0 < alpha < 255
    # compositing on the background color gives back the original (within rounding)
    comp = np.asarray(out).astype(float)
    al = comp[:, :, 3:] / 255.0
    back = comp[:, :, :3] * al + bg * (1 - al)
    assert np.abs(back[210, 100] - [253, 160, 200]).max() < 4  # numpy is [y, x]
    assert np.abs(back[100, 100] - [255, 20, 147]).max() < 2


def test_bg_color_returns_border_color():
    im = Image.new("RGB", (100, 100), (253, 190, 213))
    ImageDraw.Draw(im).rectangle((20, 20, 80, 80), fill=(0, 0, 0))
    assert pipeline.bg_color(im) == (253, 190, 213)


def test_decontaminate_removes_background_from_soft_edge():
    # A red shape cut out of a white background: the model keeps the original RGB, so
    # half-transparent edge pixels are still 50 % white. Un-premultiplying against the
    # background must give back pure red without touching alpha or opaque pixels.
    cut = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    cut.putpixel((5, 5), (220, 30, 30, 255))          # interior
    cut.putpixel((5, 6), (237, 142, 142, 128))        # 50 % red + 50 % white, alpha 128
    cut.putpixel((5, 7), (250, 225, 225, 20))         # faint fringe, almost all white
    out = pipeline.decontaminate(cut, (255, 255, 255))
    assert out.getpixel((5, 5)) == (220, 30, 30, 255)
    r, g, b, a = out.getpixel((5, 6))
    assert a == 128 and abs(r - 220) <= 3 and abs(g - 30) <= 3 and abs(b - 30) <= 3
    assert out.getpixel((5, 7))[3] == 20               # alpha never changes
    # the model gives interior pixels alpha ~250, not 255: those are not blends, leave them alone
    cut.putpixel((6, 5), (200, 100, 50, 250))
    assert pipeline.decontaminate(cut, (255, 255, 255)).getpixel((6, 5)) == (200, 100, 50, 250)
    assert out.getpixel((0, 0)) == (0, 0, 0, 0)        # transparent stays transparent


def test_detect_style_thin_line_art_is_flat():
    # Line art: almost every opaque pixel is an edge, so the old "flat body" measure has
    # nothing to look at. Two colors total must still be flat (anime upscaler, not x4plus).
    im = Image.new("RGBA", (500, 500), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for i in range(0, 500, 25):
        d.line((0, i, 499, i + 60), fill=(240, 60, 120, 255), width=3)
        d.ellipse((i, 400 - i // 2, i + 20, 420 - i // 2), outline=(240, 60, 120, 255), width=2)
    assert pipeline.detect_style(im) == "flat"


def test_key_color_antialiased_edge_becomes_semi_transparent_design_color():
    # A pink stroke on white. An edge pixel that is 50 % pink + 50 % white must come out as
    # pink at ~50 % alpha, not as an opaque pale pixel (that prints a light fringe on dark shirts).
    white = np.array([255, 255, 255]); pink = np.array([240, 60, 120])
    im = Image.new("RGB", (200, 200), tuple(white))
    d = ImageDraw.Draw(im)
    d.rectangle((50, 50, 150, 150), fill=tuple(pink))
    blend = tuple((0.5 * pink + 0.5 * white).round().astype(int))
    d.rectangle((151, 50, 151, 150), fill=blend)             # 1-px anti-aliased column
    faint = tuple((0.15 * pink + 0.85 * white).round().astype(int))
    d.rectangle((152, 50, 152, 150), fill=faint)             # 15 % coverage
    out = pipeline.key_color(im)
    r, g, b, a = out.getpixel((151, 100))
    assert 100 <= a <= 156, a
    assert abs(r - 240) <= 12 and abs(g - 60) <= 12 and abs(b - 120) <= 12, (r, g, b)
    r, g, b, a = out.getpixel((152, 100))
    assert 15 <= a <= 60, a                                  # faint coverage stays faint, not dropped
    assert out.getpixel((100, 100)) == (240, 60, 120, 255)   # solid stroke untouched
    assert out.getpixel((10, 10))[3] == 0


def test_key_color_multicolor_design_keeps_saturated_pixels_opaque():
    # Hot pink glow next to a black bolt on light pink: the pink is a design color, not a blend
    # of black with the background, so it must stay fully opaque even though a black neighbour is
    # much farther from the background. Only pixels touching the background get edge treatment,
    # and only when the recovered color stays in gamut.
    bg = (253, 190, 213); hot = (255, 20, 147); pale = (250, 120, 180)
    im = Image.new("RGB", (300, 300), bg)
    d = ImageDraw.Draw(im)
    d.rectangle((60, 60, 240, 240), fill=hot)             # hot pink area
    d.rectangle((100, 60, 110, 240), fill=(0, 0, 0))      # black bolt through it
    d.rectangle((150, 150, 200, 200), fill=pale)          # pale patch deep inside, not touching bg
    out = pipeline.key_color(im)
    assert out.getpixel((98, 150)) == hot + (255,)        # hot pink beside black: untouched
    assert out.getpixel((112, 150)) == hot + (255,)
    assert out.getpixel((61, 150)) == hot + (255,)        # solid edge pixel touching bg: still opaque
    assert out.getpixel((175, 175))[3] == 255             # pale interior is design, not a blend
    # compositing back on the background must reproduce the original everywhere
    comp = np.asarray(out).astype(float); a = comp[:, :, 3:] / 255
    back = comp[:, :, :3] * a + np.array(bg) * (1 - a)
    assert np.abs(back - np.asarray(im)).max() < 4
