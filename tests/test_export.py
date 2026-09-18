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


def test_save_print_png_records_how_the_file_was_made(tmp_path):
    """Mỗi file in tự trả lời 'chạy cờ gì', thay cho việc ghi chú tay sau mỗi lô."""
    p = tmp_path / "a.png"
    meta = {"flags": {"fill_holes": True, "scale": 26.0}, "bg": "black", "mode": "black",
            "how": "key nền đen + thân hình đặc", "cmd": "./run.sh --fill-holes --scale 26"}
    pipeline.save_print_png(Image.new("RGBA", (40, 40), (255, 0, 0, 255)), p, meta=meta)
    assert pipeline.read_meta(p) == meta


def test_read_meta_is_none_for_a_file_made_before_this(tmp_path):
    p = tmp_path / "cu.png"
    Image.new("RGBA", (4, 4)).save(p)
    assert pipeline.read_meta(p) is None


def _strokes():
    """Một khối 100 px, một nét 3 px (0,25 mm ở 300 DPI) và hai đốm 3 x 3 px."""
    import numpy as np

    a = np.zeros((300, 300, 4), np.uint8)
    a[20:120, 20:120] = (200, 30, 30, 255)          # khối: in tốt
    a[150:153, 20:280] = (30, 30, 200, 255)         # nét mảnh: DTF bong
    a[200:203, 40:43] = (30, 200, 30, 255)          # đốm
    a[200:203, 60:63] = (30, 200, 30, 255)
    return Image.fromarray(a, "RGBA")


def test_fine_ink_measures_thin_strokes_and_specks():
    import pipeline

    f = pipeline.fine_ink(_strokes())
    total = 100 * 100 + 3 * 260 + 2 * 9
    assert abs(f["manh"] - 100 * (3 * 260 + 18) / total) < 1.5   # nét và đốm đều mảnh hơn 0,5 mm
    assert abs(f["dom"] - 100 * 18 / total) < 0.1                 # chỉ hai đốm là rời và nhỏ hơn 1 mm²
    assert pipeline.fine_ink(Image.new("RGBA", (10, 10), (0, 0, 0, 0))) == {"manh": 0.0, "dom": 0.0}


def test_min_feature_thickens_thin_ink_with_its_own_color():
    import numpy as np
    import pipeline

    out = pipeline.min_feature(_strokes(), mm=0.5)
    a = np.asarray(out)
    assert pipeline.fine_ink(out)["manh"] < 1.0
    assert a[148, 100, 3] == 255 and tuple(a[148, 100, :3]) == (30, 30, 200)   # nét nới ra, đúng màu nét
    assert a[20:120, 20:120, 3].min() == 255 and tuple(a[60, 60, :3]) == (200, 30, 30)  # khối không đổi
    assert a[10, 10, 3] == 0                                                   # nền vẫn trong suốt
