import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import pipeline


def test_parse_args_defaults():
    a = pipeline.parse_args([])
    assert a.vector is False and a.colors is None and a.size == "4500x5100" and a.bg == "auto" and a.style == "auto"
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
    out = Image.open(pipeline.OUTPUT_DIR / "ok_1181x1000_center.png")
    assert out.size == (1181, 1000)
    assert round(out.info["dpi"][0]) == 300
    assert (pipeline.INPUT_DIR / "done" / "ok.png").exists()
    assert (pipeline.INPUT_DIR / "failed" / "bad.png").exists()
    assert (pipeline.REVIEW_DIR / "ok_1181x1000_center.png").exists()
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

    assert pipeline.main(["--size", "240x240", "--keep-input", "--fill-holes",
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

    # mực in đè lên áo cùng màu: bật --fill-holes nhầm cho poster
    hard = v(dict(good, thua=60.0))
    assert hard["muc"] == "hong" and "fill-holes" in hard["why"][0]

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
    out = dirs / "output" / "hinh_240x240_center.png"
    rep = pipeline.measure_print(out, src)
    assert rep["dac"] > 0 and rep["phu_thap"] >= 0
    assert rep["thua"] is None and rep["sai_so"] is None
    assert rep["shirt"] is None                     # áo khác màu nền: không biết là màu gì
    assert pipeline.print_verdict(rep)["muc"] != "hong"

    capsys.readouterr()
    assert pipeline.main(["--audit"]) == 0
    table = capsys.readouterr().out
    assert "hinh_240x240_center.png" in table and "—" in table


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
