import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

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
    assert out.getpixel((210, 100)) == (128, 128, 128, 255)  # a flat grey block is solid ink
    assert out.getpixel((100, 210)) == (0, 120, 200, 255)    # so is a flat blue one
    # compositing the result on black gives back the original color (within rounding + lo offset)
    comp = np.asarray(out).astype(float)
    back = comp[:, :, :3] * comp[:, :, 3:] / 255.0
    assert np.abs(back[210, 100] - [0, 120, 200]).max() < 16  # numpy is [y, x]: the blue block
    assert np.abs(back[100, 210] - [128, 128, 128]).max() < 16  # and here the grey one


def test_key_black_without_solid_lift_is_the_plain_brightness_key():
    out = pipeline.key_bg(_dark_art(), "black", solid=False)
    r, g, b, a = out.getpixel((210, 100))                    # grey -> white at ~half alpha
    assert (r, g, b) == (255, 255, 255) and 100 < a < 150
    r, g, b, a = out.getpixel((100, 210))                    # blue: un-premultiplied, max channel 255
    assert b == 255 and a > 180


def test_key_black_leaves_a_fade_thinning_into_the_shirt():
    """A glow is not a flat area: its alpha must keep following brightness, or it prints as a
    solid halo instead of dissolving into the shirt."""
    im = Image.new("RGB", (300, 300), (0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((120, 120, 180, 180), fill=(60, 90, 255))      # core
    im = im.filter(ImageFilter.GaussianBlur(30))             # ...and its glow
    d = ImageDraw.Draw(im)
    d.rectangle((20, 20, 80, 80), fill=(60, 90, 255))        # a flat block of the same color
    out = pipeline.key_bg(im, "black")
    assert out.getpixel((50, 50)) == (60, 90, 255, 255)      # the block: solid ink
    a_mid = out.getpixel((150, 205))[3]                      # out in the glow, well off the core
    assert 0 < a_mid < 200, a_mid                            # still a fade, not lifted to opaque
    comp = np.asarray(out).astype(float)
    back = comp[:, :, :3] * comp[:, :, 3:] / 255.0
    assert np.abs(back - np.asarray(im).astype(float)).max() < 16


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


def _pink_character():
    """Light pink bg; a character = black outline ring whose inside is painted in the bg pink
    (skin) with a slightly-off-pink patch (cheek), plus a pink glow outside the outline."""
    bg = (253, 190, 213)
    im = Image.new("RGB", (300, 300), bg)
    d = ImageDraw.Draw(im)
    d.ellipse((60, 60, 240, 240), fill=(0, 0, 0))        # outline ring...
    d.ellipse((80, 80, 220, 220), fill=bg)               # ...whose inside is the bg color
    d.rectangle((130, 130, 170, 170), fill=(253, 160, 200))  # cheek: near bg -> partial alpha when keyed
    d.rectangle((0, 250, 300, 300), fill=(254, 205, 222))    # faint glow strip touching the border
    return im


def test_solid_core_keeps_silhouette_opaque_keys_the_rest():
    im = _pink_character()
    keyed = pipeline.key_bg(im, "color")
    assert keyed.getpixel((150, 100))[3] == 0            # plain key: skin punched through
    assert 0 < keyed.getpixel((150, 150))[3] < 255       # cheek half keyed
    silhouette = Image.new("L", im.size, 0)
    ImageDraw.Draw(silhouette).ellipse((58, 58, 242, 242), fill=255)  # model mask, 2 px too generous
    out = pipeline.solid_core(keyed, im, silhouette, pipeline.bg_rgb(im, "color"))
    assert out.getpixel((150, 100)) == (253, 190, 213, 255)  # skin back, original color, opaque
    assert out.getpixel((150, 150)) == (253, 160, 200, 255)  # cheek fully opaque, original color
    assert out.getpixel((150, 70)) == (0, 0, 0, 255)         # outline untouched
    assert out.getpixel((10, 10))[3] == 0                    # outside bg still transparent
    assert out.getpixel((150, 275))[3] == keyed.getpixel((150, 275))[3]  # glow outside silhouette stays keyed
    assert out.getpixel((150, 59))[3] == 0                   # bg the model over-included is not kept


def test_solid_core_leaves_the_frame_block_to_the_key():
    """Figure cut off by the frame: the model's mask runs to the bottom border although the
    picture has already faded out there. That block must not become solid."""
    bg = (0, 0, 0)
    im = Image.new("RGB", (300, 300), bg)
    d = ImageDraw.Draw(im)
    d.rectangle((100, 40, 200, 200), fill=(30, 60, 220))        # jersey
    d.rectangle((120, 100, 180, 120), fill=(4, 6, 12))          # deep fold shadow inside the jersey
    for y in range(200, 240):                                  # hem fading into the background
        v = int(220 * (240 - y) / 40)
        d.line((100, y, 200, y), fill=(v // 7, v // 4, v))
    keyed = pipeline.key_bg(im, "black")
    silhouette = Image.new("L", im.size, 0)
    ImageDraw.Draw(silhouette).rectangle((98, 38, 202, 299), fill=255)  # model: body all the way down
    out = pipeline.solid_core(keyed, im, silhouette, pipeline.bg_rgb(im, "black"))
    assert out.getpixel((150, 110)) == (4, 6, 12, 255)          # fold shadow: solid ink
    assert out.getpixel((150, 280))[3] == 0                     # body the model invented below the picture: gone
    assert out.getpixel((150, 232))[3] == 255                   # the hem itself, dim but present: solid ink
    assert out.getpixel((150, 60)) == (30, 60, 220, 255)


def test_solid_core_composites_back_to_the_original_on_the_shirt():
    """Every pixel, figure and glow alike, must still reproduce the original on the shirt color."""
    im = _pink_character()
    keyed = pipeline.key_bg(im, "color")
    silhouette = Image.new("L", im.size, 0)
    ImageDraw.Draw(silhouette).ellipse((58, 58, 242, 242), fill=255)
    out = pipeline.solid_core(keyed, im, silhouette, pipeline.bg_rgb(im, "color"))

    def on_shirt(img):
        x = np.asarray(img).astype(float)
        a = x[:, :, 3:] / 255.0
        return x[:, :, :3] * a + np.array([253, 190, 213]) * (1 - a)

    src = np.asarray(im.convert("RGB")).astype(float)
    # making the figure opaque must not change one printed pixel: no darker ring inside the
    # edge, no brighter rim outside it. Whatever the plain key already gets wrong, no more.
    assert np.abs(on_shirt(out) - src).max() <= np.abs(on_shirt(keyed) - src).max()


def test_solid_core_ignores_a_hairline_of_mask_around_a_thin_detail():
    """A model tracing a bright thin stroke leaves a filament of mask hugging it. Filling that
    in turns the background caught inside the filament into solid dark ink."""
    im = Image.new("RGB", (300, 300), (0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle((40, 40, 160, 260), fill=(30, 60, 220))   # the figure
    d.line((200, 40, 240, 260), fill=(255, 255, 255), width=5)  # a signature stroke
    keyed = pipeline.key_bg(im, "black")
    sil = Image.new("L", im.size, 0)
    sd = ImageDraw.Draw(sil)
    sd.rectangle((40, 40, 160, 260), fill=255)
    sd.line((200, 40, 240, 260), fill=255, width=17)      # model traced the stroke: a filament
    out = pipeline.solid_core(keyed, im, sil, pipeline.bg_rgb(im, "black"))
    assert out.getpixel((100, 150)) == (30, 60, 220, 255)          # figure still filled in
    assert out.getpixel((222, 150))[3] == 255                      # the stroke itself stays
    a = np.asarray(out)
    beside = a[150, 208:214]                                       # black gap inside the filament
    assert beside[:, 3].max() < 40, f"nền trong sợi mask bị ép đặc: {beside}"


def test_key_color_floor_drops_a_near_background_area_but_keeps_edges():
    """A large area a shade off the background composites back to itself at any alpha, so it
    looks right on screen either way -- but a printer lays white underbase by alpha, and a
    third-opacity area prints as a haze. The floor drops it without touching anything above."""
    bg = (0, 0, 0)
    im = Image.new("RGB", (300, 300), bg)
    d = ImageDraw.Draw(im)
    d.ellipse((40, 40, 260, 260), fill=(15, 13, 13))       # near-black plate behind the art
    d.rectangle((90, 90, 210, 210), fill=(216, 13, 25))    # the art itself
    d.rectangle((211, 90, 211, 210), fill=(108, 7, 13))    # a 50 % anti-aliased column
    off = pipeline.key_color(im, floor=0.0)
    on = pipeline.key_color(im)                            # default floor
    assert off.getpixel((60, 150))[3] > 40                 # without a floor the plate takes ink
    assert on.getpixel((60, 150))[3] == 0                  # with it the shirt shows instead
    assert on.getpixel((150, 150)) == (216, 13, 25, 255)   # the art is untouched
    assert on.getpixel((211, 150))[3] == off.getpixel((211, 150))[3]  # so is its soft edge
    assert on.getpixel((211, 150))[:3] == off.getpixel((211, 150))[:3]


def test_solid_lift_keeps_the_anti_aliased_edge():
    """Lifting a flat area to opaque must carry its rim up by the same factor, not assign full
    alpha to it: a rim pixel at 9 % coverage slammed to opaque is a hard, bloated, speckled edge."""
    im = Image.new("RGB", (300, 300), (0, 0, 0))
    d = ImageDraw.Draw(im)
    red = (195, 20, 25)
    d.rectangle((60, 60, 240, 240), fill=red)
    for i, frac in enumerate((0.75, 0.45, 0.15)):            # a 3 px anti-aliased ramp
        v = tuple(round(c * frac) for c in red)
        d.rectangle((241 + i, 60, 241 + i, 240), fill=v)
    plain = pipeline.key_bg(im, "black", solid=False)
    out = pipeline.key_bg(im, "black")
    assert out.getpixel((150, 150))[3] >= 250                # flat interior: solid ink now
    assert plain.getpixel((150, 150))[3] < 200               # it was not before
    ramp = [out.getpixel((241 + i, 150))[3] for i in range(3)]
    assert ramp[0] > ramp[1] > ramp[2] > 0, ramp             # the rim is still a ramp
    assert ramp[0] < 250, ramp                               # and none of it went opaque
    a_plain = np.asarray(plain)[:, :, 3]
    a_out = np.asarray(out)[:, :, 3]
    assert (a_out > 0).sum() == (a_plain > 0).sum()          # no ink invented anywhere
    assert not ((a_plain <= 8) & (a_out > 200)).any()        # nothing faint jumped to opaque


def test_redundant_ink_counts_only_opaque_pixels_that_match_the_shirt():
    bg = (0, 0, 0)
    a = np.zeros((100, 100, 4), np.uint8)
    a[10:40, :, 3] = 255                      # đục và đen: in đè lên áo đen, vô ích
    a[40:70, :, :3] = 255
    a[40:70, :, 3] = 255                      # đục và trắng: mực thật
    a[70:90, :, 3] = 50                       # mờ và đen: key vốn để cho áo hiện ra, không tính
    img = Image.fromarray(a, "RGBA")
    share = pipeline.redundant_ink(img, Image.new("RGB", (100, 100)), bg)
    assert 36 < share < 39, share             # 3000 trên 8000 pixel có mực
    assert pipeline.redundant_ink(Image.new("RGBA", (10, 10), (0, 0, 0, 0)),
                                  Image.new("RGB", (10, 10)), bg) == 0.0


def test_one_ink_maps_the_shirt_to_ink_axis_into_coverage():
    """Tách một màu là chuyển độ đậm nhạt thành độ phủ, không phải bôi cả hình thành một khối."""
    shirt, ink = (255, 255, 255), (0, 0, 0)
    a = np.zeros((1, 4, 4), np.uint8)
    a[0, 0] = (255, 255, 255, 255)      # đúng màu áo -> không có mực
    a[0, 1] = (0, 0, 0, 255)            # đúng màu mực -> phủ kín
    a[0, 2] = (128, 128, 128, 255)      # nửa đường -> nửa độ phủ
    a[0, 3] = (0, 0, 0, 0)              # ngoài thiết kế -> vẫn trống
    out = np.asarray(pipeline.one_ink(Image.fromarray(a, "RGBA"), ink, shirt))
    assert out[0, 0, 3] == 0
    assert out[0, 1, 3] == 255
    assert 120 < out[0, 2, 3] < 136, out[0, 2, 3]
    assert out[0, 3, 3] == 0
    assert {tuple(c) for c in out[:, :, :3].reshape(-1, 3)} == {ink}   # đúng một màu mực


def test_one_ink_reads_the_design_as_it_sits_on_the_shirt():
    """Màu trong file đã chia ngược cho alpha, nên phải hỏi bản ghép trên áo mới ra đúng độ đậm."""
    shirt, ink = (0, 0, 0), (255, 255, 255)
    a = np.array([[[255, 255, 255, 128]]], np.uint8)   # trắng ở nửa alpha = xám trên áo đen
    out = np.asarray(pipeline.one_ink(Image.fromarray(a, "RGBA"), ink, shirt))
    assert 120 < out[0, 0, 3] < 136, out[0, 0, 3]      # nửa độ phủ, không phải phủ kín


def test_one_ink_refuses_a_color_that_would_be_invisible():
    with pytest.raises(pipeline.EmptyResult, match="không thấy gì"):
        pipeline.one_ink(Image.new("RGBA", (4, 4), (255, 255, 255, 255)), (10, 10, 10), (0, 0, 0))


def test_solid_core_fill_floor_leaves_near_background_pixels_to_the_shirt():
    """Ảnh đen trắng trên nền đen: quần và bóng sâu gần như cùng màu nền. Với sàn, chỗ đó để cho
    áo làm màu đen thay vì tô một khối mực đen lên vải đen; chỗ sáng hơn nền rõ vẫn tô đặc."""
    bg = (0, 0, 0)
    im = Image.new("RGB", (300, 300), bg)
    d = ImageDraw.Draw(im)
    d.rectangle((100, 40, 200, 200), fill=(30, 60, 220))        # áo đấu: sáng, phải đặc
    d.rectangle((120, 100, 180, 120), fill=(4, 6, 12))          # bóng sâu: cách nền 14
    d.rectangle((100, 200, 200, 260), fill=(14, 14, 14))        # quần gần đen: cách nền 24
    keyed = pipeline.key_bg(im, "black")
    silhouette = Image.new("L", im.size, 0)
    ImageDraw.Draw(silhouette).rectangle((98, 38, 202, 262), fill=255)
    bg_rgb = pipeline.bg_rgb(im, "black")

    default = pipeline.solid_core(keyed, im, silhouette, bg_rgb)
    assert default.getpixel((150, 110))[3] == 255 and default.getpixel((150, 230))[3] == 255  # như cũ: tô hết

    floored = pipeline.solid_core(keyed, im, silhouette, bg_rgb, floor=32.0)
    assert floored.getpixel((150, 60)) == (30, 60, 220, 255)     # áo vẫn đặc, đúng màu
    assert floored.getpixel((150, 110))[3] == keyed.getpixel((150, 110))[3]   # bóng sâu: về alpha key
    assert floored.getpixel((150, 230))[3] == keyed.getpixel((150, 230))[3]   # quần gần đen: về alpha key
    assert floored.getpixel((150, 230))[3] < 60
    assert pipeline.parse_args([]).fill_floor == 0.0
    assert pipeline.parse_args(["--fill-floor", "32"]).fill_floor == 32.0


def test_fill_floor_and_guard_anchor_on_the_measured_background():
    """Nền ảnh AI hiếm khi đen tuyệt đối: (19, 19, 19) là thường. Sàn và chốt chặn phải đo khoảng
    cách tới màu nền ĐO ĐƯỢC, không phải tới (0, 0, 0), nếu không thân người xám 27 mức lọt qua."""
    bg = (19, 19, 19)
    im = Image.new("RGB", (300, 300), bg)
    d = ImageDraw.Draw(im)
    d.rectangle((100, 40, 200, 120), fill=(230, 230, 230))     # áo trắng
    d.rectangle((100, 120, 200, 260), fill=(27, 27, 27))       # quần: cách nền đo được 14, cách (0,0,0) 47
    keyed = pipeline.key_bg(im, "black")
    silhouette = Image.new("L", im.size, 0)
    ImageDraw.Draw(silhouette).rectangle((98, 38, 202, 262), fill=255)
    out = pipeline.solid_core(keyed, im, silhouette, pipeline.bg_rgb(im, "black"), floor=32.0)
    assert out.getpixel((150, 80))[3] == 255                   # áo trắng đặc
    assert out.getpixel((150, 200))[3] == keyed.getpixel((150, 200))[3] < 60   # quần để cho áo
    full = pipeline.solid_core(keyed, im, silhouette, pipeline.bg_rgb(im, "black"))
    assert pipeline.redundant_ink(full, im, pipeline.bg_color(im)) > 40   # đo đúng nền: khối quần là mực trùng áo
    assert pipeline.redundant_ink(full, im, (0, 0, 0)) < 5              # đo sai nền (0,0,0) thì không thấy gì
