import pytest
import numpy as np
from PIL import Image, ImageDraw

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


def test_refine_edge_trusts_core_keys_band_recovers_thin_detail():
    pink = (253, 190, 213)
    im = Image.new("RGB", (300, 300), pink)
    d = ImageDraw.Draw(im)
    d.rectangle((60, 60, 240, 240), fill=(0, 0, 0))       # design body
    d.rectangle((72, 120, 180, 180), fill=pink)           # bg-colored patch reaching the core edge: is design
    d.rectangle((240, 148, 285, 152), fill=(0, 0, 0))     # thin line the model will drop
    # fake model mask: body grown by 20 px (bg included), thin line missed
    model = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    body = im.crop((40, 40, 261, 261)).convert("RGBA")
    model.paste(body, (40, 40))

    out = pipeline.refine_edge(model, im.convert("RGBA"), band=0.1)  # r = 30 px

    a = np.asarray(out)[:, :, 3]  # numpy is [y, x]
    assert a[150, 150] == 255          # bg-colored patch deep inside the core stays opaque
    assert 0 < a[150, 75] < 255        # ...but feathers out toward the core edge (no hard seam)
    assert a[45, 45] == 0              # bg inside the model mask but in the band -> transparent
    assert a[150, 270] == 255          # thin line outside the mask recovered
    assert a[5, 5] == 0 and a[150, 295] == 0  # far background untouched


def test_dilate_grows_mask_without_wrapping():
    m = np.zeros((20, 20), dtype=bool)
    m[0, 0] = True
    out = pipeline._dilate(m, 2)
    assert out[2, 2] and not out[3, 3]
    assert not out[-1, -1] and not out[0, -1]  # no wrap-around to the far edge


def test_refine_edge_does_not_grow_model_holes():
    # Line art: the model leaves the white gaps between strokes open. Shrinking the core must
    # only happen from the outer silhouette, never from those holes, or every gap widens into
    # a ring of missing stroke. Gaps themselves stay transparent (they are background-colored).
    white = (255, 255, 255)
    im = Image.new("RGB", (600, 600), white)
    d = ImageDraw.Draw(im)
    d.rectangle((40, 40, 560, 560), fill=(200, 120, 40))     # body
    d.rectangle((250, 250, 350, 350), fill=white)            # white gap inside the body
    d.rectangle((100, 400, 500, 424), fill=white)            # narrow white slit, thinner than the feather
    model = Image.new("RGBA", (600, 600), (0, 0, 0, 0))
    model.paste(im.crop((40, 40, 561, 561)).convert("RGBA"), (40, 40))
    a = np.asarray(model).copy()
    a[250:351, 250:351, 3] = 0      # the model left the gap open (correct)
    a[400:425, 100:501, 3] = 0      # and the slit too
    model = Image.fromarray(a, "RGBA")

    out = np.asarray(pipeline.refine_edge(model, im.convert("RGBA"), band=0.1))[:, :, 3]  # r = 60

    assert out[300, 300] == 0        # gap stays open
    assert out[300, 245] == 255      # body right next to the gap keeps full alpha (hole not grown)
    assert out[412, 300] == 0        # the slit stays open too: core feather must not leak across it
    assert out[100, 100] == 255 and out[10, 10] == 0


def test_art_box_scales_the_canvas_both_ways():
    assert pipeline.art_box((4500, 5100), 100) == (4500, 5100)
    assert pipeline.art_box((4500, 5100), 26) == (1170, 1326)   # the placement in a real print file
    assert pipeline.art_box((4500, 5100), 0.01) == (1, 1)       # never degenerate


def test_place_on_canvas_anchors_with_a_margin(red_square):
    box = (1000, 1200)
    art = red_square.crop(red_square.getbbox())                 # 100 x 100
    m = round(min(box) * 0.02)                                  # default margin: 2 % of short side

    def corner(placed):
        ys, xs = np.nonzero(np.asarray(placed)[:, :, 3] > 0)
        return xs.min(), ys.min(), xs.max(), ys.max()

    assert corner(pipeline.place_on_canvas(art, box)) == (450, 550, 549, 649)   # centered as before
    assert corner(pipeline.place_on_canvas(art, box, "top-right")) == (
        box[0] - m - 100, m, box[0] - m - 1, m + 99)
    assert corner(pipeline.place_on_canvas(art, box, "bottom-left")) == (
        m, box[1] - m - 100, m + 99, box[1] - m - 1)
    assert corner(pipeline.place_on_canvas(art, box, "top")) == (450, m, 549, m + 99)  # centered across
    assert pipeline.place_on_canvas(art, box, "top-right").size == box


def test_place_on_canvas_margin_is_configurable(red_square):
    art = red_square.crop(red_square.getbbox())
    tight = pipeline.place_on_canvas(art, (1000, 1200), "top-left", margin=0.0)
    assert np.asarray(tight)[0, 0, 3] > 0                       # flush into the corner
