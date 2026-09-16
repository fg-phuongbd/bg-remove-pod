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


def test_clean_print_drops_invisible_ink_but_keeps_halftone():
    """Hạt halftone và vệt sờn cố ý cũng nhỏ, nhưng chúng đậm; bụi thì vừa nhỏ vừa mờ."""
    import numpy as np

    a = np.zeros((200, 200, 4), np.uint8)
    a[50:150, 50:150, 3] = 255                 # thân thiết kế
    a[10, 10, 3] = 6                           # pixel gần như vô hình
    a[20:23, 20:23, 3] = 25                    # đốm nhỏ và mờ = bụi
    a[30:33, 30:33, 3] = 240                   # hạt halftone: nhỏ nhưng đậm
    out = np.asarray(pipeline.clean_print(Image.fromarray(a, "RGBA")))[:, :, 3]
    assert out[10, 10] == 0
    assert out[21, 21] == 0
    assert out[31, 31] == 240, "hạt halftone đậm phải giữ nguyên"
    assert out[100, 100] == 255


def test_save_print_png_tags_srgb_and_dpi(tmp_path):
    p = tmp_path / "a.png"
    pipeline.save_print_png(Image.new("RGBA", (40, 40), (255, 0, 0, 255)), p)
    im = Image.open(p)
    assert im.info["icc_profile"], "RIP cần hồ sơ màu, không có thì nó tự đoán"
    assert round(im.info["dpi"][0]) == pipeline.DPI


def test_save_print_png_can_skip_the_cleanup(tmp_path):
    import numpy as np

    a = np.zeros((20, 20, 4), np.uint8)
    a[:, :, 3] = 5
    pipeline.save_print_png(Image.fromarray(a, "RGBA"), tmp_path / "b.png", clean=False)
    assert np.asarray(Image.open(tmp_path / "b.png"))[:, :, 3].max() == 5
