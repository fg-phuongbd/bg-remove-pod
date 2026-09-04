from PIL import Image

import pipeline


def test_save_print_png(tmp_path, red_circle):
    p = tmp_path / "o.png"
    pipeline.save_print_png(red_circle, p)
    dpi = Image.open(p).info["dpi"]
    assert round(dpi[0]) == 300 and round(dpi[1]) == 300


def test_make_review(tmp_path, red_circle):
    p = tmp_path / "r.png"
    pipeline.make_review(red_circle, red_circle.resize((200, 400)), p)
    im = Image.open(p)
    assert im.height == 800
    assert im.width == 800 + 20 + 400
