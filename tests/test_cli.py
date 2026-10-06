import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

import pipeline


def test_parse_args_defaults():
    a = pipeline.parse_args([])
    assert a.vector is False and a.colors is None and a.size == "4500x5400" and a.bg == "auto" and a.style == "auto"
    assert a.keep_input is False and a.files == []


def test_parse_args_flags():
    a = pipeline.parse_args(["--vector", "--colors", "6", "--size", "21x29.7", "--keep-input", "a.png"])
    assert a.vector and a.colors == 6 and a.size == "21x29.7" and a.keep_input and a.files == ["a.png"]


@pytest.mark.skipif(shutil.which("resvg") is None, reason="resvg not installed")
def test_main_batch_isolates_failures(tmp_path, monkeypatch, red_circle):
    for name in ("INPUT_DIR", "OUTPUT_DIR", "REVIEW_DIR", "WORK_DIR"):
        d = tmp_path / name.lower()
        d.mkdir()
        monkeypatch.setattr(pipeline, name, d)
    monkeypatch.setattr(pipeline, "remove_bg", lambda img: img)
    monkeypatch.setattr(pipeline, "check_tools", lambda: None)
    red_circle.save(pipeline.INPUT_DIR / "ok.png")
    (pipeline.INPUT_DIR / "bad.png").write_text("not an image")

    rc = pipeline.main(["--vector", "--size", "1181x1000"])

    assert rc == 1
    out = Image.open(pipeline.OUTPUT_DIR / "ok_1181x1000_center_vector.png")   # --vector để dấu trong tên
    assert out.size == (1181, 1000)
    assert round(out.info["dpi"][0]) == 300
    assert (pipeline.INPUT_DIR / "done" / "ok.png").exists()
    assert (pipeline.INPUT_DIR / "failed" / "bad.png").exists()
    assert (pipeline.REVIEW_DIR / "ok_1181x1000_center_vector.png").exists()
    assert not (pipeline.INPUT_DIR / "ok.png").exists()


def test_parse_args_fill_holes():
    assert pipeline.parse_args(["--fill-holes"]).fill_holes is True
    assert pipeline.parse_args([]).fill_holes is False


def test_parse_args_bg_color():
    assert pipeline.parse_args(["--bg", "color"]).bg == "color"


def test_parse_args_shirt():
    assert pipeline.parse_args([]).shirt == "same"   # default: the shirt is the background color
    assert pipeline.parse_args(["--shirt", "auto"]).shirt == "auto"
    assert pipeline.parse_args(["--shirt", "same"]).shirt == "same"
    assert pipeline.parse_args(["--shirt", "other"]).shirt == "other"
    with pytest.raises(SystemExit):
        pipeline.parse_args(["--refine"])  # folded into --shirt / auto detection


def test_bg_override_beats_shirt():
    # --bg is the hidden escape hatch: when given, it wins over --shirt
    assert pipeline.parse_args(["--bg", "color"]).bg == "color"
    assert pipeline.choose_mode(pipeline.parse_args(["--bg", "ai", "--shirt", "same"]), "color") == ("ai", True)
    assert pipeline.choose_mode(pipeline.parse_args(["--shirt", "same"]), "color") == ("color", False)
    assert pipeline.choose_mode(pipeline.parse_args([]), "black") == ("black", False)
    assert pipeline.choose_mode(pipeline.parse_args([]), "white") == ("white", False)   # default same: key, no model
    assert pipeline.choose_mode(pipeline.parse_args([]), "color") == ("color", False)
    assert pipeline.choose_mode(pipeline.parse_args(["--shirt", "auto"]), "white") == ("ai", True)


@pytest.mark.skipif(shutil.which("resvg") is None, reason="resvg not installed")
def test_main_keyed_black_art_makes_review_on_shirt(tmp_path, monkeypatch):
    for name in ("INPUT_DIR", "OUTPUT_DIR", "REVIEW_DIR", "WORK_DIR"):
        d = tmp_path / name.lower()
        d.mkdir()
        monkeypatch.setattr(pipeline, name, d)
    monkeypatch.setattr(pipeline, "check_tools", lambda: None)
    monkeypatch.setattr(pipeline, "upscale", lambda img, scale=4, model="": img.resize((img.width * scale, img.height * scale)))
    art = Image.new("RGB", (200, 200), (10, 10, 12))
    art.paste((240, 40, 40), (50, 50, 150, 150))
    art.save(pipeline.INPUT_DIR / "dark.png")

    assert pipeline.main(["--size", "400x400"]) == 0  # auto: black bg -> keyed for a dark shirt

    assert (pipeline.OUTPUT_DIR / "dark_400x400_center.png").exists()
    review = Image.open(pipeline.REVIEW_DIR / "dark_400x400_center.png").convert("RGB")
    assert review.getpixel((review.width - 3, 3)) == (20, 20, 22)  # result shown on the dark shirt color


def test_key_style_picks_distance_for_flat_art():
    k = pipeline.key_style
    assert k("black", True) == "color"    # solid ink wants distance, not brightness
    assert k("white", True) == "color"
    assert k("black", False) == "black"   # glows and photos keep the brightness key
    assert k("white", False) == "white"
    assert k("color", True) == "color"    # already the distance key
    assert k("color", False) == "color"


def test_parse_args_placement_defaults():
    a = pipeline.parse_args([])
    assert (a.place, a.scale, a.margin) == ("center", 100.0, 2.0)
    b = pipeline.parse_args(["--place", "top-right", "--scale", "26"])
    assert (b.place, b.scale) == ("top-right", 26.0)


def test_out_name_carries_canvas_and_placement():
    n = pipeline.out_name
    assert n("skull", (4500, 5100), "center", 100.0) == "skull_4500x5100_center.png"
    assert n("skull", (4500, 5100), "top-right", 26.0) == "skull_4500x5100_top-right_26pc.png"
    # cùng một góc, hai cỡ khác nhau thì không đè lên nhau
    assert n("skull", (4500, 5100), "top-right", 40.0) != n("skull", (4500, 5100), "top-right", 26.0)


def test_parse_args_fill_limit():
    assert pipeline.parse_args([]).fill_limit == pipeline.FILL_LIMIT
    assert pipeline.parse_args(["--fill-limit", "100"]).fill_limit == 100.0


def _poster_on_black(path):
    """Poster: hình sáng rải rác trên nền đen, có quầng sáng mờ giữa chúng.

    Quầng đó là thứ làm chốt chặn cần thiết: nó nằm trên ngưỡng nhiễu nên key để lại alpha nhỏ,
    thoát khỏi các chốt sẵn có trong solid_core, rồi bị tô thành mực đen đặc. Nền đen tuyệt đối
    thì solid_core đã tự chặn được."""
    from PIL import ImageDraw, ImageFilter
    im = Image.new("RGB", (240, 240), (0, 0, 0))
    d = ImageDraw.Draw(im)
    for box in ((20, 20, 90, 60), (150, 20, 220, 60), (20, 180, 220, 220)):
        d.rectangle(box, fill=(240, 240, 240))
    glow = Image.new("L", (240, 240), 0)
    ImageDraw.Draw(glow).rectangle((30, 30, 210, 210), fill=255)
    g = np.asarray(glow.filter(ImageFilter.GaussianBlur(28))).astype(float) / 255.0
    a = np.asarray(im).astype(float) + g[:, :, None] * np.array([16, 14, 20])
    a[:6] = a[-6:] = 0
    a[:, :6] = a[:, -6:] = 0                       # vành biên vẫn đen tuyệt đối để đo đúng nhiễu nền
    Image.fromarray(a.round().clip(0, 255).astype(np.uint8), "RGB").save(path)


def test_fill_holes_is_skipped_when_it_would_print_over_the_shirt(tmp_path, monkeypatch, capsys):
    """Model cắt hình coi cả tấm poster là một khối; tô đặc sẽ biến nền đen thành mực đen."""
    for name, attr in (("input", "INPUT_DIR"), ("output", "OUTPUT_DIR"),
                       ("work", "WORK_DIR"), ("review", "REVIEW_DIR")):
        d = tmp_path / name
        d.mkdir()
        monkeypatch.setattr(pipeline, attr, d)
    monkeypatch.setattr(pipeline, "check_tools", lambda: None)
    monkeypatch.setattr(pipeline, "upscale", lambda img, scale=4, model="": img)
    # silhouette phủ kín tấm ảnh, đúng như model làm với poster
    monkeypatch.setattr(pipeline, "remove_bg",
                        lambda img: Image.new("RGBA", img.size, (255, 255, 255, 255)))
    src = tmp_path / "input" / "poster.png"
    _poster_on_black(src)

    assert pipeline.main(["--size", "240x240", "--keep-input", "--fill-holes", "--fill-limit", "20",
                          "--bg", "black", str(src)]) == 0
    log = capsys.readouterr().out
    assert "BỎ QUA --fill-holes" in log
    assert "thân hình đặc" not in log            # dòng log phải nói đúng việc đã làm
    out = np.asarray(Image.open(tmp_path / "output" / "poster_240x240_center.png"))
    gap = out[110:160, 110:130, 3]               # khoảng đen giữa các hình
    assert gap.max() < 40, "nền giữa các chi tiết phải để cho áo hiện ra"

    assert pipeline.main(["--size", "240x240", "--keep-input", "--fill-holes", "--bg", "black",
                          "--fill-limit", "100", str(src)]) == 0
    assert "BỎ QUA" not in capsys.readouterr().out   # tắt chốt chặn thì vẫn tô như cũ


def test_parse_ink_accepts_names_and_hex():
    assert pipeline.parse_ink("black") == (0, 0, 0)
    assert pipeline.parse_ink("WHITE") == (255, 255, 255)
    assert pipeline.parse_ink("#ff0055") == (255, 0, 85)
    assert pipeline.parse_ink("none") is None and pipeline.parse_ink("") is None
    for bad in ("xanh", "#fff", "#gggggg"):
        with pytest.raises(Exception):
            pipeline.parse_ink(bad)


def test_out_name_records_a_non_default_ink():
    n = pipeline.out_name
    assert n("a", (4500, 5100), "center", 100.0, "none") == "a_4500x5100_center.png"
    assert n("a", (4500, 5100), "center", 100.0, "black") == "a_4500x5100_center_ink-black.png"
    assert n("a", (4500, 5100), "center", 100.0, "#ff0055") == "a_4500x5100_center_ink-ff0055.png"


def test_load_image_applies_the_exif_rotation(tmp_path):
    """Ảnh chụp điện thoại lưu nằm ngang kèm cờ xoay; đọc thô thì file in ra sai hướng."""
    import io
    im = Image.new("RGB", (400, 200), (0, 0, 0))
    im.paste(Image.new("RGB", (100, 100), (255, 0, 0)), (0, 0))
    exif = Image.Exif()
    exif[274] = 6                                     # Orientation: xoay 90 độ
    p = tmp_path / "nghieng.jpg"
    im.save(p, "JPEG", exif=exif)
    assert Image.open(p).size == (400, 200)           # thô: vẫn nằm ngang
    assert pipeline.load_image(p).size == (200, 400)  # đã xoay đúng
    assert pipeline.load_image(p).mode == "RGBA"


def test_audit_prints_a_table_and_flags_what_to_check(tmp_path, monkeypatch, capsys):
    for name, attr in (("input", "INPUT_DIR"), ("output", "OUTPUT_DIR"),
                       ("work", "WORK_DIR"), ("review", "REVIEW_DIR")):
        d = tmp_path / name
        d.mkdir()
        monkeypatch.setattr(pipeline, attr, d)
    monkeypatch.setattr(pipeline, "check_tools", lambda: None)
    monkeypatch.setattr(pipeline, "upscale", lambda img, scale=4, model="": img)
    src = tmp_path / "input" / "hinh.png"
    _poster_on_black(src)
    assert pipeline.main(["--size", "240x240", "--keep-input", str(src)]) == 0
    capsys.readouterr()

    assert pipeline.main(["--audit"]) == 0
    out = capsys.readouterr().out
    assert "hinh_240x240_center.png" in out
    assert "mực đặc tb" in out and "1 file" in out

    # một file toàn mực trùng màu áo phải bị đánh dấu
    bad = np.zeros((40, 40, 4), np.uint8)
    bad[..., 3] = 255
    Image.fromarray(bad, "RGBA").save(tmp_path / "output" / "hinh_240x240_top-left.png")
    assert pipeline.main(["--audit"]) == 0
    out = capsys.readouterr().out
    assert "Không dùng được" in out and "--fill-holes" in out
    assert "đủ điều kiện in:" in out and "/2 file" in out


def test_audit_says_so_when_there_is_nothing_to_grade(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(pipeline, "INPUT_DIR", tmp_path / "input")
    (tmp_path / "input").mkdir()
    assert pipeline.main(["--audit"]) == 0
    assert "Chưa có file in nào" in capsys.readouterr().out


def test_ink_is_refused_when_it_matches_the_real_background(tmp_path, monkeypatch, capsys):
    """Màu áo dùng để xem là (20, 20, 22) cho dễ nhìn ra vải; phép tách một màu phải hỏi nền thật."""
    for name, attr in (("input", "INPUT_DIR"), ("output", "OUTPUT_DIR"),
                       ("work", "WORK_DIR"), ("review", "REVIEW_DIR")):
        d = tmp_path / name
        d.mkdir()
        monkeypatch.setattr(pipeline, attr, d)
    monkeypatch.setattr(pipeline, "check_tools", lambda: None)
    monkeypatch.setattr(pipeline, "upscale", lambda img, scale=4, model="": img)
    src = tmp_path / "input" / "toi.png"
    _poster_on_black(src)

    assert pipeline.main(["--size", "240x240", "--keep-input", "--ink", "black", str(src)]) == 1
    assert "không thấy gì" in capsys.readouterr().out
    assert not list((tmp_path / "output").glob("*.png"))
    assert src.exists(), "--keep-input phải giữ ảnh lại để đổi màu mực rồi chạy lại"

    assert pipeline.main(["--size", "240x240", "--keep-input", "--ink", "white", str(src)]) == 0
    out = np.asarray(Image.open(tmp_path / "output" / "toi_240x240_center_ink-white.png"))
    ink = out[:, :, 3] > 0
    assert ink.any()
    assert {tuple(c) for c in out[:, :, :3][ink].reshape(-1, 3)} == {(255, 255, 255)}


def test_keep_input_leaves_a_failed_image_where_it_is(tmp_path, monkeypatch):
    """Trên trang, ảnh lỗi bị chuyển sang failed/ sẽ biến mất khỏi danh sách và hết đường sửa."""
    for name, attr in (("input", "INPUT_DIR"), ("output", "OUTPUT_DIR"),
                       ("work", "WORK_DIR"), ("review", "REVIEW_DIR")):
        d = tmp_path / name
        d.mkdir()
        monkeypatch.setattr(pipeline, attr, d)
    monkeypatch.setattr(pipeline, "check_tools", lambda: None)
    (tmp_path / "input" / "hong.png").write_text("không phải ảnh")

    assert pipeline.main(["--keep-input"]) == 1
    assert (tmp_path / "input" / "hong.png").exists()
    assert not (tmp_path / "input" / "failed").exists()

    assert pipeline.main([]) == 1
    assert (tmp_path / "input" / "failed" / "hong.png").exists()


def test_low_coverage_warns_about_dtf(tmp_path, monkeypatch, capsys):
    """Vùng dưới 40% độ phủ nhận ít bột keo khi in DTF, nên phải được báo trước."""
    import numpy as np

    for name, attr in (("input", "INPUT_DIR"), ("output", "OUTPUT_DIR"),
                       ("work", "WORK_DIR"), ("review", "REVIEW_DIR")):
        d = tmp_path / name
        d.mkdir()
        monkeypatch.setattr(pipeline, attr, d)
    monkeypatch.setattr(pipeline, "check_tools", lambda: None)
    monkeypatch.setattr(pipeline, "upscale", lambda img, scale=4, model="": img)

    # Nền đen, một lõi sáng và quầng sáng rộng quanh nó. Quầng mới là thứ cho ra mực phủ thấp:
    # một mảng xám phẳng thì bước nâng mảng màu khối sẽ đưa lên đặc, đúng như nó phải làm.
    from PIL import ImageDraw, ImageFilter
    im = Image.new("RGB", (240, 240), (0, 0, 0))
    ImageDraw.Draw(im).ellipse((80, 80, 160, 160), fill=(255, 255, 255))
    im = im.filter(ImageFilter.GaussianBlur(18))
    src = tmp_path / "input" / "mong.png"
    im.save(src)

    assert pipeline.main(["--size", "240x240", "--keep-input", "--bg", "black", str(src)]) == 0
    assert "CẢNH BÁO in DTF" in capsys.readouterr().out

    assert pipeline.main(["--size", "240x240", "--keep-input", "--bg", "black",
                          "--dtf-warn", "100", str(src)]) == 0
    assert "CẢNH BÁO in DTF" not in capsys.readouterr().out


def test_low_coverage_reads_the_print_file():
    import numpy as np

    a = np.zeros((10, 10, 4), np.uint8)
    a[0:5, :, 3] = 255          # đặc
    a[5:8, :, 3] = 60           # dưới 40% độ phủ
    img = Image.fromarray(a, "RGBA")
    assert 35 < pipeline.low_coverage(img) < 40      # 30 trên 80 pixel có mực
    assert pipeline.low_coverage(Image.new("RGBA", (4, 4), (0, 0, 0, 0))) == 0.0


def test_print_verdict_is_the_single_place_that_decides():
    v = pipeline.print_verdict
    good = {"dac": 95.0, "phu_thap": 0.4, "thua": 0.1, "sai_so": 0.3}
    assert v(good)["muc"] == "dat" and v(good)["why"] == []

    # phủ thấp là rủi ro của DTF, in được nhưng nên thử giặt trước
    soft = v(dict(good, phu_thap=15.7))
    assert soft["muc"] == "xem" and "DTF" in soft["why"][0]

    # mực in đè lên áo cùng màu mà không ai xin tô đặc: có gì đó sai
    hard = v(dict(good, thua=60.0))
    assert hard["muc"] == "hong" and "fill-holes" in hard["why"][0]
    # cùng con số nhưng file được cố ý tô đặc thân hình: đó là ý người dùng, chỉ ghi nhận
    meant = v(dict(good, thua=60.0, fill=True))
    assert meant["muc"] == "dat" and meant["why"] == [] and meant["info"]

    assert v(dict(good, sai_so=9.0))["muc"] == "hong"
    assert v({"dac": 0.0, "phu_thap": 0.0, "thua": 0.0, "sai_so": 0.0})["muc"] == "hong"
    assert v(dict(good, phu_thap=15.7), dtf_warn=100)["muc"] == "dat"   # ngưỡng đổi được


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """input/output/work/review riêng cho một test, không đụng ảnh thật; bỏ qua tool ngoài."""
    for name, attr in (("input", "INPUT_DIR"), ("output", "OUTPUT_DIR"),
                       ("work", "WORK_DIR"), ("review", "REVIEW_DIR")):
        d = tmp_path / name
        d.mkdir()
        monkeypatch.setattr(pipeline, attr, d)
    monkeypatch.setattr(pipeline, "check_tools", lambda: None)
    monkeypatch.setattr(pipeline, "upscale", lambda img, scale=4, model="": img)
    return tmp_path


def test_audit_survives_an_empty_print_file(dirs, capsys):
    """Một file in không có pixel mực nào phải hiện là 'file rỗng', không được làm sập cả bảng."""
    src = dirs / "input" / "hinh.png"
    _poster_on_black(src)
    Image.new("RGBA", (40, 40), (0, 0, 0, 0)).save(dirs / "output" / "hinh_240x240_center.png")
    assert pipeline.main(["--audit"]) == 0
    out = capsys.readouterr().out
    assert "hinh_240x240_center.png" in out
    assert "file rỗng" in out and "Không dùng được" in out


def test_flags_used_keeps_only_what_differs_from_the_defaults():
    a = pipeline.parse_args(["--fill-holes", "--scale", "26", "--place", "top-right", "--keep-input", "x.png"])
    assert pipeline.flags_used(a) == {"fill_holes": True, "scale": 26.0, "place": "top-right"}
    assert pipeline.flags_used(pipeline.parse_args([])) == {}
    assert pipeline.cmd_line({"fill_holes": True, "scale": 26.0, "place": "top-right"}) == \
        "./run.sh --fill-holes --scale 26 --place top-right"
    assert pipeline.cmd_line({}) == "./run.sh"


def test_print_file_carries_its_flags_and_the_chosen_method(dirs):
    src = dirs / "input" / "hinh.png"
    _poster_on_black(src)
    assert pipeline.main(["--size", "240x240", "--keep-input", "--bg", "black", str(src)]) == 0
    meta = pipeline.read_meta(dirs / "output" / "hinh_240x240_center.png")
    assert meta["flags"] == {"size": "240x240", "bg": "black"}
    assert meta["bg"] == "black" and meta["mode"] == "black"
    assert meta["how"] == "key nền đen"
    assert meta["cmd"] == "./run.sh --size 240x240 --bg black"


def _fake_cutout(img):
    """Thay model cắt hình: pixel tối trên nền đen thành trong suốt, còn lại giữ đục."""
    a = np.asarray(img.convert("RGBA")).copy()
    a[:, :, 3] = np.where(a[:, :, :3].max(axis=2) > 60, 255, 0)
    return Image.fromarray(a, "RGBA")


def test_measure_does_not_grade_a_cutout_against_the_wrong_shirt(dirs, monkeypatch, capsys):
    """--shirt other là in lên áo KHÁC màu nền, nên 'mực trùng màu áo' và 'sai số' so với màu nền
    gốc là con số vô nghĩa, và có thể kết luận 'Không dùng được' nhầm. Không đo được thì nói không
    đo được, đừng bịa số."""
    monkeypatch.setattr(pipeline, "remove_bg", _fake_cutout)
    src = dirs / "input" / "hinh.png"
    _poster_on_black(src)
    assert pipeline.main(["--size", "240x240", "--keep-input", "--shirt", "other", str(src)]) == 0
    out = dirs / "output" / "hinh_240x240_center_cutout.png"                  # áo khác màu để dấu trong tên
    rep = pipeline.measure_print(out, src)
    assert rep["dac"] > 0 and rep["phu_thap"] >= 0
    assert rep["thua"] is None and rep["sai_so"] is None
    assert rep["shirt"] is None                     # áo khác màu nền: không biết là màu gì
    assert pipeline.print_verdict(rep)["muc"] != "hong"

    capsys.readouterr()
    assert pipeline.main(["--audit"]) == 0
    table = capsys.readouterr().out
    assert "hinh_240x240_center_cutout.png" in table and "—" in table


def test_measure_skips_fidelity_for_a_one_ink_file(dirs):
    """Tách một màu cố ý khác ảnh gốc, nên 'sai số khi in' không có nghĩa; mực trùng màu áo thì vẫn đo."""
    src = dirs / "input" / "toi.png"
    _poster_on_black(src)
    assert pipeline.main(["--size", "240x240", "--keep-input", "--ink", "white", str(src)]) == 0
    rep = pipeline.measure_print(dirs / "output" / "toi_240x240_center_ink-white.png", src)
    assert rep["sai_so"] is None
    assert rep["thua"] is not None and rep["thua"] < 5
    assert rep["shirt"] == "#141416"


def test_print_verdict_copes_with_unmeasured_columns():
    v = pipeline.print_verdict({"dac": 90.0, "phu_thap": 1.0, "thua": None, "sai_so": None, "shirt": None})
    assert v["muc"] == "dat"
    v = pipeline.print_verdict({"dac": 0.0, "phu_thap": 0.0, "thua": None, "sai_so": None, "shirt": None})
    assert v["muc"] == "hong" and "rỗng" in v["why"][0]


def test_preset_fills_place_and_scale_unless_given_explicitly():
    """`--preset chest-left` thay cho việc nhớ 'top-right 26'. Cờ đặt tay vẫn thắng preset."""
    a = pipeline.apply_preset(pipeline.parse_args(["--preset", "chest-left"]))
    assert (a.place, a.scale) == ("top-right", 26.0)     # ngực trái người mặc = bên phải file
    a = pipeline.apply_preset(pipeline.parse_args(["--preset", "chest-left", "--scale", "30"]))
    assert (a.place, a.scale) == ("top-right", 30.0)
    a = pipeline.apply_preset(pipeline.parse_args([]))
    assert (a.place, a.scale) == ("center", 100.0)
    assert set(pipeline.PRESETS) >= {"full", "chest-left", "chest-right", "chest", "back-neck"}
    with pytest.raises(SystemExit):
        pipeline.parse_args(["--preset", "khong-co"])


def test_preset_shows_in_the_file_name_and_is_remembered(dirs):
    src = dirs / "input" / "hinh.png"
    _poster_on_black(src)
    assert pipeline.main(["--size", "240x240", "--keep-input", "--preset", "chest-left", str(src)]) == 0
    out = dirs / "output" / "hinh_240x240_top-right_26pc.png"
    assert out.exists()
    assert pipeline.read_meta(out)["cmd"] == "./run.sh --size 240x240 --preset chest-left"


def test_warns_when_the_source_is_too_small_for_the_print_size(dirs, capsys):
    """Model upscale được 4 lần; ảnh 240 px lên khung 1200 px là 5 lần, phần dư là kéo giãn."""
    src = dirs / "input" / "nho.png"
    _poster_on_black(src)
    assert pipeline.main(["--size", "1200x1200", "--keep-input", str(src)]) == 0
    log = capsys.readouterr().out
    assert "CẢNH BÁO ảnh gốc nhỏ" in log and "5,0 lần" in log

    assert pipeline.main(["--size", "240x240", "--keep-input", str(src)]) == 0
    assert "CẢNH BÁO ảnh gốc nhỏ" not in capsys.readouterr().out


def test_grainy_art_is_upscaled_with_lanczos(dirs, monkeypatch, capsys):
    """Halftone qua Real-ESRGAN thành vệt lông xù; ảnh hạt phải đi đường Lanczos."""
    import subprocess
    calls = []
    monkeypatch.setattr(pipeline, "upscale", lambda img, scale=4, model="": (calls.append(model), img)[1])
    from PIL import ImageDraw
    im = Image.new("RGB", (240, 240), (0, 0, 0))
    d = ImageDraw.Draw(im)
    for y in range(30, 210, 6):                     # chừa lề: viền ảnh phải là nền để nhận diện được
        for x in range(30, 210, 6):
            d.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(240, 240, 240))
    src = dirs / "input" / "cham.png"
    im.save(src)
    assert pipeline.main(["--size", "240x240", "--keep-input", str(src)]) == 0
    assert "kiểu: grain" in capsys.readouterr().out
    assert calls == ["lanczos"]
    assert pipeline.parse_args(["--style", "grain"]).style == "grain"


def test_verdict_and_audit_carry_fine_ink(dirs, capsys):
    from tests.test_export import _strokes
    src = dirs / "input" / "net.png"
    im = Image.new("RGB", (240, 240), (0, 0, 0))          # nguồn sạch: sai số key phải gần 0
    im.paste((215, 8, 22), (60, 60, 180, 180))
    im.save(src)
    out = dirs / "output" / "net_300x300_center.png"
    pipeline.save_print_png(_strokes(), out, clean=False)
    rep = pipeline.measure_print(out, src)
    assert rep["manh"] > 5 and rep["dom"] > 0
    v = pipeline.print_verdict(rep)
    assert v["muc"] == "xem" and any("mảnh" in w for w in v["why"])
    assert pipeline.main(["--audit"]) == 0
    assert "mảnh" in capsys.readouterr().out


def test_dtf_safe_thickens_before_saving(dirs):
    from PIL import ImageDraw
    im = Image.new("RGB", (240, 240), (0, 0, 0))
    ImageDraw.Draw(im).line((20, 120, 220, 120), fill=(240, 240, 240), width=1)   # nét 1 px
    src = dirs / "input" / "line.png"
    im.save(src)
    assert pipeline.main(["--size", "240x240", "--keep-input", str(src)]) == 0
    thin = pipeline.fine_ink(Image.open(dirs / "output" / "line_240x240_center.png"))["manh"]
    assert pipeline.main(["--size", "240x240", "--keep-input", "--dtf-safe", str(src)]) == 0
    safe = pipeline.fine_ink(Image.open(dirs / "output" / "line_240x240_center_dtf-safe.png"))["manh"]
    assert thin > 50 and safe < 5


def test_out_name_tags_the_flags_that_change_the_picture():
    """Hai lần chạy khác cờ xử lý phải ra hai file, không đè nhau."""
    n = pipeline.out_name
    base = n("a", (4500, 5100), "center", 100.0)
    assert base == "a_4500x5100_center.png"
    assert n("a", (4500, 5100), "center", 100.0, fill=True) == "a_4500x5100_center_fill.png"
    assert n("a", (4500, 5100), "center", 100.0, cutout=True) == "a_4500x5100_center_cutout.png"
    assert n("a", (4500, 5100), "center", 100.0, vector=True) == "a_4500x5100_center_vector.png"
    assert n("a", (4500, 5100), "center", 100.0, style="grain") == "a_4500x5100_center_grain.png"
    assert n("a", (4500, 5100), "center", 100.0, style="auto") == base          # auto = không ép, không đuôi
    assert n("a", (4500, 5100), "center", 100.0, colors=8) == "a_4500x5100_center_c8.png"
    # thứ tự cố định để cùng bộ cờ luôn ra cùng tên
    assert n("a", (4500, 5100), "top-right", 26.0, "black", True, fill=True, cutout=True) == \
        "a_4500x5100_top-right_26pc_ink-black_cutout_fill_dtf-safe.png"


def test_two_runs_with_different_flags_keep_both_files(dirs, monkeypatch):
    monkeypatch.setattr(pipeline, "remove_bg", _fake_cutout)
    src = dirs / "input" / "hinh.png"
    _poster_on_black(src)
    assert pipeline.main(["--size", "240x240", "--keep-input", str(src)]) == 0
    assert pipeline.main(["--size", "240x240", "--keep-input", "--fill-holes", "--fill-limit", "100", str(src)]) == 0
    assert pipeline.main(["--size", "240x240", "--keep-input", "--shirt", "other", str(src)]) == 0
    names = sorted(p.name for p in (dirs / "output").glob("hinh_*.png"))
    assert names == ["hinh_240x240_center.png", "hinh_240x240_center_cutout.png", "hinh_240x240_center_fill.png"]
    assert pipeline.read_meta(dirs / "output" / "hinh_240x240_center_fill.png")["flags"]["fill_holes"] is True


def test_fill_that_is_skipped_does_not_tag_the_file(dirs, monkeypatch, capsys):
    """Chốt chặn bỏ qua tô đặc thì file không được mang đuôi _fill, vì nó không hề được tô."""
    monkeypatch.setattr(pipeline, "remove_bg", lambda img: Image.new("RGBA", img.size, (255, 255, 255, 255)))
    src = dirs / "input" / "poster.png"
    _poster_on_black(src)
    assert pipeline.main(["--size", "240x240", "--keep-input", "--fill-holes", "--fill-limit", "20",
                          "--bg", "black", str(src)]) == 0
    assert "BỎ QUA --fill-holes" in capsys.readouterr().out
    assert (dirs / "output" / "poster_240x240_center.png").exists()
    assert not (dirs / "output" / "poster_240x240_center_fill.png").exists()


def test_verdict_explains_redundant_ink_without_blaming_a_poster():
    v = pipeline.print_verdict({"dac": 97.0, "phu_thap": 0.5, "thua": 36.0, "sai_so": 1.0, "shirt": "#141416"})
    assert v["muc"] == "hong"
    assert "không bật thân hình đặc" in v["why"][0] and "poster" in v["why"][0]


def test_fill_guard_measures_the_print_resolution_image(dirs, monkeypatch, capsys):
    """Chốt chặn đo ở ảnh gốc từng cho qua một tấm mà file in cuối có 36% mực trùng áo: model
    upscale làm mịn vùng tối về sát màu nền. Phải đo trên chính bản sẽ in."""
    from PIL import ImageDraw
    im = Image.new("RGB", (240, 240), (0, 0, 0))
    ImageDraw.Draw(im).rectangle((60, 40, 180, 220), fill=(40, 40, 40))     # thân người xám tối, cách nền 69
    src = dirs / "input" / "toi.png"
    im.save(src)
    monkeypatch.setattr(pipeline, "remove_bg", lambda img: Image.new("RGBA", img.size, (255, 255, 255, 255)))

    def smoothing_upscale(img, scale=4, model=""):          # "upscale" kéo nửa dưới thân người về sát màu nền
        a = np.asarray(img.convert("RGB")).copy()
        body = a.max(axis=2) > 0
        body[:130] = False
        a[body] = (12, 12, 12)
        return Image.fromarray(a, "RGB").convert(img.mode)
    monkeypatch.setattr(pipeline, "upscale", smoothing_upscale)

    assert pipeline.main(["--size", "240x240", "--keep-input", "--fill-holes", "--fill-limit", "20",
                          "--bg", "black", str(src)]) == 0
    log = capsys.readouterr().out
    assert "BỎ QUA --fill-holes" in log
    assert (dirs / "output" / "toi_240x240_center.png").exists()
    assert not (dirs / "output" / "toi_240x240_center_fill.png").exists()
    rep = pipeline.measure_print(dirs / "output" / "toi_240x240_center.png", src)
    assert rep["thua"] < 20


def test_upscale_style_does_not_depend_on_fill_holes(dirs, monkeypatch, capsys):
    """Cùng một ảnh phải ra cùng kiểu upscale dù có tô đặc hay không: kiểu đo trên bức ảnh,
    không đo trên việc thân hình đã được lấp hay chưa. Trên ảnh thật, một tấm đổi từ flat sang
    grain chỉ vì bật cờ, và bản tô đặc bị mềm đi vì đi đường Lanczos."""
    from PIL import ImageDraw
    rng = np.random.default_rng(1)
    a = np.zeros((240, 240, 3), np.uint8)
    body = rng.random((120, 120)) < 0.5                                 # thân người: xám tối lẫn gần đen, hạt nhiễu
    a[30:150, 60:180] = np.where(body[:, :, None], (40, 40, 40), (12, 12, 12))
    a[30:60, 60:180] = (230, 230, 230)                                 # vai áo trắng: chắc chắn là mực đặc
    im = Image.fromarray(a, "RGB")
    src = dirs / "input" / "nguoi.png"
    im.save(src)
    sil = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(sil).rectangle((58, 28, 182, 152), fill=(255, 255, 255, 255))
    monkeypatch.setattr(pipeline, "remove_bg", lambda img: sil)

    def style_of(flags):
        assert pipeline.main(["--size", "240x240", "--keep-input", "--bg", "black", "--fill-limit", "100",
                              *flags, str(src)]) == 0
        line = [l for l in capsys.readouterr().out.splitlines() if "kiểu:" in l][0]
        return line.split("kiểu:")[1].split("|")[0].strip()

    styles = {"không cờ": style_of([]), "tô đặc": style_of(["--fill-holes"]),
              "sàn 32": style_of(["--fill-holes", "--fill-floor", "32"])}
    assert len(set(styles.values())) == 1, f"kiểu đổi theo cờ: {styles}"


def test_warnings_are_recorded_in_the_print_file(dirs, monkeypatch, capsys):
    """Người dùng trang không thấy terminal. Mọi cảnh báo của một lần chạy phải đi theo file in
    để trang hiện lại được: chốt chặn bỏ qua tô đặc, phủ thấp DTF, ảnh gốc nhỏ."""
    monkeypatch.setattr(pipeline, "remove_bg", lambda img: Image.new("RGBA", img.size, (255, 255, 255, 255)))
    src = dirs / "input" / "poster.png"
    _poster_on_black(src)
    assert pipeline.main(["--size", "1200x1200", "--keep-input", "--fill-holes", "--fill-limit", "20",
                          "--bg", "black", str(src)]) == 0
    log = capsys.readouterr().out
    notes = pipeline.read_meta(dirs / "output" / "poster_1200x1200_center.png")["notes"]
    assert any(n.startswith("BỎ QUA --fill-holes") for n in notes)
    assert any(n.startswith("CẢNH BÁO ảnh gốc nhỏ") for n in notes)
    for n in notes:
        assert n in log                                   # cùng câu chữ với terminal, không hai phiên bản

    src2 = dirs / "input" / "sach.png"
    im = Image.new("RGB", (240, 240), (0, 0, 0))
    im.paste((215, 8, 22), (60, 60, 180, 180))
    im.save(src2)
    assert pipeline.main(["--size", "240x240", "--keep-input", str(src2)]) == 0
    assert pipeline.read_meta(dirs / "output" / "sach_240x240_center.png")["notes"] == []


def test_fill_limit_is_off_by_default():
    """Người dùng chỉ bật thân hình đặc cho ảnh có người, và muốn người là một khối đặc đúng như
    model cắt, kể cả quần đen trên áo đen. Chốt chặn không được cản điều đó; ai muốn thì tự bật
    bằng --fill-limit 20."""
    assert pipeline.FILL_LIMIT == 100.0
    assert pipeline.parse_args([]).fill_limit == 100.0


def test_fill_holes_covers_a_dark_figure_by_default(dirs, monkeypatch, capsys):
    """Ảnh đen trắng, thân người gần màu nền: bật thân hình đặc thì phải tô hết, không hỏi lại."""
    from PIL import ImageDraw
    im = Image.new("RGB", (240, 240), (19, 19, 19))
    d = ImageDraw.Draw(im)
    d.rectangle((60, 40, 180, 120), fill=(230, 230, 230))       # áo trắng
    d.rectangle((60, 120, 180, 220), fill=(27, 27, 27))         # quần gần đen
    src = dirs / "input" / "toi.png"
    im.save(src)
    sil = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(sil).rectangle((58, 38, 182, 222), fill=(255, 255, 255, 255))
    monkeypatch.setattr(pipeline, "remove_bg", lambda img: sil)
    assert pipeline.main(["--size", "240x240", "--keep-input", "--fill-holes", str(src)]) == 0
    assert "BỎ QUA" not in capsys.readouterr().out
    out = dirs / "output" / "toi_240x240_center_fill.png"
    assert np.asarray(Image.open(out))[170, 120, 3] == 255      # quần: mực đặc, đúng ý người dùng
    rep = pipeline.measure_print(out, src)
    assert rep["thua"] > 20 and rep["fill"] is True
    v = pipeline.print_verdict(rep)
    assert v["muc"] == "dat", v                                  # cố ý tô đặc thì không phải lỗi
    assert any("tô đặc" in w for w in v.get("info", [])), v      # nhưng vẫn nói cho biết


def test_upscale_plan_skips_the_model_when_the_source_is_already_big_enough():
    """Ảnh 4K chỉ cần phóng khoảng 1,1 lần cho khung 4500 x 5100. Qua model 4 lần thì ra ảnh 16384 px,
    Pillow từ chối mở và máy hết RAM; Lanczos từ chi tiết thật là đủ."""
    plan = pipeline.upscale_plan
    inner = (4410, 5000)
    assert plan((4096, 4096), (3900, 4000), inner) is None       # phóng 1,25 lần: không cần model
    assert plan((4096, 4096), (2300, 2600), inner) is None       # phóng 1,9 lần: vẫn dưới ngưỡng 2
    s = plan((3840, 2160), (2100, 2100), inner)                   # 4K màn hình, hình vuông: cần 2,1 lần
    assert s is not None and (3840 * 2160 * s * s * 16) <= pipeline.MODEL_MAX_PX * 1.001  # qua model, đã thu
    assert plan((1254, 1254), (1100, 1200), inner) == 1.0        # ảnh ChatGPT: model như cũ, không thu


def test_upscale_plan_keeps_the_model_for_a_small_placement_of_a_normal_source():
    """chest-left chỉ cần ảnh 1254 px phóng 1,1 lần, nhưng model vẫn làm mép và mảng mực đặc hơn:
    bỏ model thì mực đặc từ 72% xuống 50%, phủ thấp từ 10% lên 15% trên ảnh thật."""
    assert pipeline.upscale_plan((1254, 1254), (1100, 1200), (1170, 1326)) == 1.0
    assert pipeline.upscale_plan((1024, 1536), (900, 1400), (900, 1020)) == 1.0      # back-neck, co nhỏ


def test_upscale_plan_shrinks_a_mid_size_source_so_the_model_output_fits_in_memory():
    plan = pipeline.upscale_plan
    inner = (4410, 5000)
    s = plan((2048, 2048), (2000, 2000), inner)                  # cần 2,2 lần, 4 lần thì 67 triệu pixel
    assert s is not None and s < 1
    assert (2048 * s * pipeline.UPSCALE) ** 2 <= pipeline.MODEL_MAX_PX * 1.001   # làm tròn số thực
    assert 2000 * s * pipeline.UPSCALE >= 4410                   # thu rồi vẫn đủ lấp khung, không kéo giãn


def test_upscale_plan_prefers_lanczos_over_shrinking_away_real_detail():
    """Hình nhỏ trong ảnh gốc lớn: thu ảnh cho vừa model sẽ bỏ chi tiết thật rồi kéo giãn lại."""
    assert pipeline.upscale_plan((4096, 4096), (1200, 1200), (4410, 5000)) is None


def test_a_big_source_never_reaches_the_model(dirs, monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("ảnh gốc đủ lớn thì không được đi qua model upscale")
    monkeypatch.setattr(pipeline, "upscale", boom)
    monkeypatch.setattr(pipeline, "MODEL_MAX_PX", 16 * 300 * 300)   # 400 px coi như "quá lớn cho model"
    art = Image.new("RGB", (400, 400), (0, 0, 0))
    art.paste((240, 40, 40), (20, 20, 380, 380))
    src = dirs / "input" / "big.png"
    art.save(src)

    assert pipeline.main(["--size", "600x600", "--keep-input", str(src)]) == 0   # phóng chưa tới 2 lần
    out = Image.open(dirs / "output" / "big_600x600_center.png")
    assert out.size == (600, 600)
    assert np.asarray(out)[300, 300, 3] == 255
    assert "phóng: Lanczos" in capsys.readouterr().out


def test_a_mid_size_source_is_shrunk_before_the_model(dirs, monkeypatch):
    seen = []

    def fake(img, scale=4, model=""):
        seen.append(img.size)
        return img.resize((img.width * scale, img.height * scale))
    monkeypatch.setattr(pipeline, "upscale", fake)
    monkeypatch.setattr(pipeline, "MODEL_MAX_PX", 16 * 150 * 150)   # ngân sách nhỏ cho test nhanh
    art = Image.new("RGB", (200, 200), (0, 0, 0))
    art.paste((240, 40, 40), (4, 4, 196, 196))
    src = dirs / "input" / "mid.png"
    art.save(src)

    assert pipeline.main(["--size", "500x500", "--keep-input", str(src)]) == 0
    assert seen and seen[0][0] <= 150, seen                         # vào model đã thu
    assert Image.open(dirs / "output" / "mid_500x500_center.png").size == (500, 500)


def _halftone_on_black(path, size=240):
    from PIL import ImageDraw
    im = Image.new("RGB", (size, size), (0, 0, 0))
    d = ImageDraw.Draw(im)
    for y in range(30, size - 30, 6):                 # chừa lề: viền ảnh phải là nền để nhận diện được
        for x in range(30, size - 30, 6):
            d.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(240, 240, 240))
    im.save(path)


def test_grain_art_gets_hard_dots_when_it_is_enlarged(dirs, monkeypatch, capsys):
    calls = []
    real = pipeline.harden_dots
    monkeypatch.setattr(pipeline, "harden_dots", lambda img, grow: (calls.append(grow), real(img, grow))[1])
    monkeypatch.setattr(pipeline, "upscale",
                        lambda img, scale=4, model="": img.resize((img.width * scale, img.height * scale), Image.Resampling.LANCZOS))
    src = dirs / "input" / "cham.png"
    _halftone_on_black(src)
    assert pipeline.main(["--size", "960x960", "--keep-input", str(src)]) == 0
    assert "kiểu: grain" in capsys.readouterr().out
    assert len(calls) == 1 and calls[0] > 3              # phóng khoảng 4 lần
    a = np.asarray(Image.open(dirs / "output" / "cham_960x960_center.png"))[..., 3]
    ink = a > 8
    assert (a[ink] < pipeline.DTF_COVERAGE).mean() < 0.10  # chấm đặc, không còn dải mờ dày


def test_hard_dots_are_only_for_enlarged_grain_art(dirs, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(pipeline, "harden_dots", lambda img, grow: (calls.append(grow), img)[1])
    src = dirs / "input" / "cham.png"
    _halftone_on_black(src, size=600)
    assert pipeline.main(["--size", "300x300", "--keep-input", str(src)]) == 0   # thu nhỏ: không có dốc mờ
    assert "kiểu: grain" in capsys.readouterr().out
    assert calls == []
    flat = dirs / "input" / "phang.png"
    art = Image.new("RGB", (200, 200), (0, 0, 0)); art.paste((240, 40, 40), (40, 40, 160, 160)); art.save(flat)
    assert pipeline.main(["--size", "800x800", "--keep-input", str(flat)]) == 0
    assert "kiểu: grain" not in capsys.readouterr().out
    assert calls == []                                    # đồ họa phẳng không đổi


def _photo_on_black(path):
    """Một 'ảnh chụp' người trên nền đen: thân sáng có nhiễu, giữa thân là áo tối gần màu nền."""
    rng = np.random.default_rng(5)
    a = rng.normal(14, 3, (240, 240, 3))
    yy, xx = np.mgrid[:240, :240]
    body = ((xx - 120) / 60) ** 2 + ((yy - 120) / 100) ** 2 < 1
    shade = 170 + 50 * np.sin(xx / 9.0) * np.cos(yy / 13.0)
    a[body] = (shade[body, None] + rng.normal(0, 12, (int(body.sum()), 3)))
    a[(abs(xx - 120) < 35) & (abs(yy - 120) < 50)] = 22  # áo đen: key thủng phần này
    Image.fromarray(a.clip(0, 255).astype(np.uint8), "RGB").save(path)


def test_suggests_solid_body_for_a_photo_keyed_full_of_holes(dirs, capsys):
    """Ảnh chụp người trên nền cùng màu áo mà chưa bật thân hình đặc: chỗ áo tối bị key thủng.
    Trang phải gợi ý bật thân hình đặc; bật rồi hoặc ảnh phẳng (logo, chữ) thì im lặng."""
    src = dirs / "input" / "cau_thu.png"
    _photo_on_black(src)
    assert pipeline.main(["--size", "240x240", "--keep-input", "--bg", "black", str(src)]) == 0
    notes = pipeline.read_meta(dirs / "output" / "cau_thu_240x240_center.png")["notes"]
    assert any(n.startswith("GỢI Ý thân hình đặc") for n in notes), notes

    src2 = dirs / "input" / "sach.png"
    im = Image.new("RGB", (240, 240), (0, 0, 0))
    im.paste((215, 8, 22), (60, 60, 180, 180))
    im.paste((0, 0, 0), (100, 100, 140, 140))         # lỗ cố ý trong logo phẳng
    im.save(src2)
    assert pipeline.main(["--size", "240x240", "--keep-input", str(src2)]) == 0
    notes = pipeline.read_meta(dirs / "output" / "sach_240x240_center.png")["notes"]
    assert not any(n.startswith("GỢI Ý") for n in notes)


def test_a_cutout_that_lost_design_is_recovered_and_noted(dirs, monkeypatch):
    """Model cắt hình giữ người, bỏ chữ: file in vẫn phải có chữ, và lần chạy phải nói đã lấy lại."""
    def figure_only(img):
        m = Image.new("L", img.size, 0)
        ImageDraw.Draw(m).ellipse((30, 30, 130, 170), fill=255)
        out = img.convert("RGBA"); out.putalpha(m)
        return out
    monkeypatch.setattr(pipeline, "remove_bg", figure_only)
    src = dirs / "input" / "cau_thu.png"
    im = Image.new("RGB", (300, 200), (12, 12, 12))
    d = ImageDraw.Draw(im)
    d.ellipse((30, 30, 130, 170), fill=(200, 30, 40))
    d.rectangle((180, 60, 280, 140), fill=(240, 240, 240))
    im.save(src)
    assert pipeline.main(["--size", "300x200", "--keep-input", "--shirt", "other", str(src)]) == 0
    out = dirs / "output" / "cau_thu_300x200_center_cutout.png"
    a = np.asarray(Image.open(out).convert("RGBA"))[:, :, 3]
    assert (a > 250).sum() > 0.9 * (100 * 140 * 0.785 + 100 * 80)   # cả người lẫn khối chữ đều có mực
    meta = pipeline.read_meta(out)
    assert any(n.startswith("ĐÃ LẤY LẠI") for n in meta["notes"]), meta["notes"]
    assert meta["kept"] > 99
    rep = pipeline.measure_print(out, src)
    assert rep["giu"] > 99 and pipeline.print_verdict(rep)["muc"] != "hong"


def test_verdict_fails_a_cutout_that_kept_too_little_of_the_design():
    good = {"dac": 90, "phu_thap": 1, "manh": 0, "dom": 0, "thua": None, "sai_so": None, "giu": 99.5}
    assert pipeline.print_verdict(good)["muc"] == "dat"
    bad = pipeline.print_verdict(dict(good, giu=38.0))
    assert bad["muc"] == "hong" and any("62%" in w for w in bad["why"])


def test_halftone_fade_flag_marks_the_file_and_explains_its_dots(dirs):
    src = dirs / "input" / "glow.png"
    yy, xx = np.mgrid[:240, :240]
    d = np.hypot(xx - 120, yy - 120)
    v = np.clip(255 - (d - 40) * 3, 14, 255)                # đĩa sáng, glow tan dần vào nền đen
    Image.fromarray(np.dstack([v, v, v]).astype(np.uint8), "RGB").save(src)
    assert pipeline.main(["--size", "240x240", "--keep-input", str(src)]) == 0
    assert pipeline.main(["--size", "240x240", "--keep-input", "--halftone-fade", str(src)]) == 0
    out = dirs / "output" / "glow_240x240_center_ht.png"
    assert out.exists()
    plain = pipeline.measure_print(dirs / "output" / "glow_240x240_center.png", src)
    rep = pipeline.measure_print(out, src)
    assert rep["phu_thap"] <= plain["phu_thap"]   # hiệu quả đo ở test_raster và trên ảnh thật
    assert rep["ht"] and not plain["ht"]
    v = pipeline.print_verdict(dict(rep, dom=5.0))
    assert not any("đốm rời" in w for w in v["why"])        # đốm là do bật chấm hóa: chỉ ghi nhận
    assert any("chấm halftone" in w for w in v["info"])


def _photo_with_figure(path):
    """Nền đen, 'chữ' trắng bên phải, 'người' (ellipse) bên trái; thân hình đặc sẽ hỏi model cắt hình."""
    rng = np.random.default_rng(1)
    a = rng.normal(14, 2, (240, 300, 3))
    yy, xx = np.mgrid[:240, :300]
    fig = ((xx - 90) / 60) ** 2 + ((yy - 120) / 100) ** 2 < 1
    a[fig] = 150 + rng.normal(0, 25, (int(fig.sum()), 3))
    a[40:200, 200:280] = 235
    Image.fromarray(a.clip(0, 255).astype(np.uint8), "RGB").save(path)
    m = Image.new("L", (300, 240), 0)
    ImageDraw.Draw(m).ellipse((30, 20, 150, 220), fill=255)
    return m


@pytest.mark.parametrize("shirt", ["same", "other"])
def test_figure_uses_the_photo_model_and_the_rest_keeps_the_sharp_one(dirs, monkeypatch, shirt):
    """Ảnh có người (thân hình đặc bật), kiểu ảnh chụp: trong thân người dùng model ảnh chụp cho da tự
    nhiên, phần còn lại (chữ đồ họa) giữ model sắc nét, vì model ảnh chụp làm vân chữ lốm đốm xám."""
    src = dirs / "input" / "nguoi.png"
    mask = _photo_with_figure(src)
    monkeypatch.setattr(pipeline, "remove_bg", lambda img: Image.merge("RGBA", (*img.convert("RGB").split(), mask.resize(img.size))))
    monkeypatch.setattr(pipeline, "photo_model_ready", lambda: True)
    tint = {pipeline.PHOTO_MODEL: (0, 0, 255), "realesrgan-x4plus": (255, 0, 0)}

    def fake_upscale(img, scale=4, model=""):
        big = img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)
        rgb = Image.new("RGB", big.size, tint.get(model, (0, 255, 0)))
        if big.mode == "RGBA":
            rgb.putalpha(big.getchannel("A"))
            return rgb
        return Image.composite(rgb, Image.new("RGB", big.size, (0, 0, 0)), big.convert("L").point(lambda v: 255 if v > 40 else 0))
    monkeypatch.setattr(pipeline, "upscale", fake_upscale)
    assert pipeline.main(["--size", "1200x960", "--keep-input", "--fill-holes", "--style", "detail",
                          "--shirt", shirt, str(src)]) == 0
    out = [p for p in (dirs / "output").glob("nguoi_*.png")][0]
    o = np.asarray(Image.open(out).convert("RGBA")).astype(int)
    ys, xs = np.nonzero(o[:, :, 3] > 200)
    x0, x1 = xs.min(), xs.max()
    w = x1 - x0
    body = o[o.shape[0] // 2, x0 + int(w * 0.2)]            # giữa thân người
    text = o[o.shape[0] // 2, x0 + int(w * 0.85)]           # giữa khối chữ
    assert body[2] > 150 and body[0] < 100, body             # người: model ảnh chụp (xanh)
    assert text[0] > 150 and text[2] < 100, text             # chữ: model sắc nét (đỏ)
