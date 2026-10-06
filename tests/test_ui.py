import io
import json
import sys
import urllib.parse
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline  # noqa: E402
import ui  # noqa: E402

OK = {"X-Clearcut": "1"}  # header trang gửi kèm mọi POST; trang web lạ không gửi được nếu không qua CORS


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Thư mục input/output/work riêng, để test không đụng vào ảnh thật của dự án."""
    for name, attr in (("input", "INPUT_DIR"), ("output", "OUTPUT_DIR"),
                       ("work", "WORK_DIR"), ("review", "REVIEW_DIR")):
        d = tmp_path / name
        d.mkdir()
        monkeypatch.setattr(pipeline, attr, d)
    monkeypatch.setattr(ui, "CACHE", tmp_path / "work" / "ui")
    (tmp_path / "input" / "done").mkdir()
    return tmp_path


def _design(path: Path):
    im = Image.new("RGB", (240, 240), (0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle((40, 40, 200, 200), fill=(215, 8, 22))
    d.rectangle((90, 90, 150, 150), fill=(255, 255, 255))
    im.save(path)
    return im


def test_make_args_starts_from_the_cli_defaults():
    """Trang không được lệch khỏi dòng lệnh khi có cờ mới, nên nó mượn chính mặc định của CLI."""
    a = ui.make_args({"scale": 26.0, "place": "top-right"})
    assert (a.scale, a.place) == (26.0, "top-right")
    assert a.size == pipeline.parse_args([]).size      # cờ không đụng tới giữ nguyên mặc định
    assert a.keep_input is True                        # chạy lại cùng ảnh là việc thường trên trang
    assert a.merge == pipeline.MERGE_DELTA_E           # cả những cờ trang không hiện


def test_make_args_rejects_an_unknown_flag():
    with pytest.raises(ValueError, match="dungsai"):
        ui.make_args({"dungsai": 1})


def test_measure_reads_solidity_waste_and_fidelity(workspace):
    src = workspace / "input" / "a.png"
    _design(src)
    pipeline.process_one(src, ui.make_args({"size": "600x600"}))
    out = workspace / "output" / "a_600x600_center.png"
    assert out.exists()
    rep = ui.measure(out, src)
    assert rep["dac"] > 80                             # thiết kế phẳng: mực phải đặc
    assert rep["thua"] < 5                             # không in đè lên áo đen
    assert rep["sai_so"] < 8                           # ghép lên áo ra lại ảnh gốc
    assert rep["shirt"] == "#141416"                   # nền đen -> màu áo đen để xem


def test_measure_on_an_empty_file_does_not_blow_up(workspace):
    out = workspace / "output" / "b.png"
    Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(out)
    src = workspace / "input" / "b.png"
    _design(src)
    assert ui.measure(out, src)["dac"] == 0.0


def test_list_images_sees_both_folders_and_flags_finished_work(workspace):
    _design(workspace / "input" / "cho.png")
    _design(workspace / "input" / "done" / "xong.png")
    Image.new("RGBA", (8, 8)).save(workspace / "output" / "xong_4500x5100_center.png")
    rows = {r["name"]: r for r in ui.list_images()}
    assert rows["cho.png"]["waiting"] is True and rows["cho.png"]["done"] is False
    assert rows["xong.png"]["waiting"] is False and rows["xong.png"]["done"] is True
    assert rows["xong.png"]["out"] == "xong_4500x5100_center.png"


def test_preview_is_cached_until_the_source_changes(workspace):
    src = workspace / "input" / "a.png"
    _design(src)
    first = ui.preview(src, 64, "src")
    assert max(Image.open(first).size) == 64
    stamp = first.stat().st_mtime_ns
    assert ui.preview(src, 64, "src").stat().st_mtime_ns == stamp   # dùng lại
    src.touch()
    _design(src)
    assert ui.preview(src, 64, "src").stat().st_mtime_ns != stamp   # gốc đổi thì dựng lại


@pytest.fixture
def server(workspace):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), ui.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def get(base, path):
    with urllib.request.urlopen(base + path) as r:  # noqa: S310 - địa chỉ do test dựng
        return r.status, r.read()


def test_server_serves_the_page_and_the_listing(server, workspace):
    _design(workspace / "input" / "a.png")
    status, body = get(server, "/")
    assert status == 200 and b"Clearcut" in body
    status, body = get(server, "/api/images")
    assert [r["name"] for r in json.loads(body)] == ["a.png"]
    status, body = get(server, "/src/a.png")
    assert status == 200 and Image.open(__import__("io").BytesIO(body)).mode == "RGBA"


def test_server_answers_a_missing_image_with_404(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        get(server, "/src/khong-co.png")
    assert e.value.code == 404


def test_server_runs_a_job_and_reports_on_it(server, workspace):
    src = workspace / "input" / "a.png"
    _design(src)
    body = json.dumps({"jobs": [{"name": "a.png", "settings": {"size": "400x400"}}]}).encode()
    req = urllib.request.Request(server + "/api/run", body, {"Content-Type": "application/json", **OK})
    with urllib.request.urlopen(req) as r:  # noqa: S310
        assert r.status == 200
    for _ in range(600):                                  # hàng chạy trong luồng nền
        if not json.loads(get(server, "/api/status")[1])["running"]:
            break
        __import__("time").sleep(0.1)
    st = json.loads(get(server, "/api/status")[1])
    assert st["done"] == ["a.png"], st["errors"]
    assert (workspace / "output" / "a_400x400_center.png").exists()
    # Chạy xong thì ảnh rời hàng chờ sang input/done/, như dòng lệnh; vẫn chạy lại được từ đó.
    assert not src.exists() and (workspace / "input" / "done" / "a.png").exists()
    assert [i["waiting"] for i in ui.list_images() if i["name"] == "a.png"] == [False]
    rep = json.loads(get(server, "/api/report/a.png")[1])
    assert rep["dac"] > 80
    body = json.dumps({"jobs": [{"name": "a.png", "settings": {"size": "300x300"}}]}).encode()
    urllib.request.urlopen(urllib.request.Request(server + "/api/run", body, {"Content-Type": "application/json", **OK}))  # noqa: S310
    for _ in range(600):
        if not json.loads(get(server, "/api/status")[1])["running"]:
            break
        __import__("time").sleep(0.1)
    assert (workspace / "output" / "a_300x300_center.png").exists(), "chạy lại từ done/ phải được"


def test_server_refuses_a_job_with_an_unknown_flag(server, workspace):
    _design(workspace / "input" / "a.png")
    body = json.dumps({"jobs": [{"name": "a.png", "settings": {"dungsai": 1}}]}).encode()
    req = urllib.request.Request(server + "/api/run", body, {"Content-Type": "application/json", **OK})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)  # noqa: S310
    assert e.value.code == 400
    assert not ui.RUNNER.status()["running"]


def test_outputs_for_lists_every_variant_newest_first(workspace):
    out = workspace / "output"
    for n in ("a_4500x5100_center.png", "a_4500x5100_top-right_26pc.png", "b_4500x5100_center.png"):
        Image.new("RGBA", (4, 4)).save(out / n)
    import os, time
    os.utime(out / "a_4500x5100_top-right_26pc.png", (time.time() + 5, time.time() + 5))
    names = [p.name for p in ui.outputs_for("a")]
    assert names == ["a_4500x5100_top-right_26pc.png", "a_4500x5100_center.png"]
    assert ui.outputs_for("khong-co") == []


def test_safe_name_keeps_uploads_inside_input():
    assert ui.safe_name("skull.png") == "skull.png"
    assert ui.safe_name("/tmp/evil/../skull.PNG") == "skull.PNG"
    assert ui.safe_name("C:\\Users\\a\\skull.jpg") == "skull.jpg"
    for bad in ("../../etc/passwd", "note.txt", "", ".hidden.png", "skull.svg"):
        with pytest.raises(ValueError):
            ui.safe_name(bad)


def test_server_accepts_a_dropped_image_and_refuses_junk(server, workspace):
    import io
    buf = io.BytesIO()
    _design(workspace / "tmp.png").save(buf, "PNG")
    req = urllib.request.Request(server + "/api/upload?name=" + urllib.parse.quote("moi.png"),
                                 buf.getvalue(), {"Content-Type": "application/octet-stream", **OK})
    with urllib.request.urlopen(req) as r:  # noqa: S310
        assert json.loads(r.read())["name"] == "moi.png"
    assert (workspace / "input" / "moi.png").exists()
    assert "moi.png" in [i["name"] for i in ui.list_images()]

    bad = urllib.request.Request(server + "/api/upload?name=moi2.png", b"khong phai anh",
                                 {"Content-Type": "application/octet-stream", **OK})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(bad)  # noqa: S310
    assert e.value.code == 400


def test_server_serves_the_print_file_as_a_download(server, workspace):
    src = workspace / "input" / "a.png"
    _design(src)
    pipeline.process_one(src, ui.make_args({"size": "300x300"}))
    with urllib.request.urlopen(server + "/file/a.png") as r:  # noqa: S310
        assert "a_300x300_center.png" in r.headers["Content-Disposition"]
        assert Image.open(io.BytesIO(r.read())).size == (300, 300)


def test_runner_runs_jobs_in_parallel(workspace, monkeypatch):
    """Nhiều ảnh chạy cùng lúc, không phải lần lượt."""
    import time
    peak = [0]
    lock = threading.Lock()
    live = [0]

    def slow(src, args):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.4)
        with lock:
            live[0] -= 1

    monkeypatch.setattr(pipeline, "process_one", slow)
    for n in ("a.png", "b.png", "c.png", "d.png"):
        _design(workspace / "input" / n)
    r = ui.Runner()
    r.start([(n, {}) for n in ("a.png", "b.png", "c.png", "d.png")], workers=3)
    for _ in range(200):
        if not r.status()["running"]:
            break
        time.sleep(0.05)
    st = r.status()
    assert sorted(st["done"]) == ["a.png", "b.png", "c.png", "d.png"], st["errors"]
    assert peak[0] == 3, f"chạy song song tối đa {peak[0]} thay vì 3"


def test_page_config_covers_every_flag_the_cli_has(server):
    """Trang phải có đủ cờ của dòng lệnh, và mặc định phải đọc từ chính parse_args."""
    cfg = json.loads(get(server, "/api/config")[1])
    names = {f["name"] for f in cfg["flags"]}
    cli = set(vars(pipeline.parse_args([]))) - pipeline.RUN_ONLY      # cờ của một lần chạy, không của ảnh
    assert names == cli, f"lệch: {cli ^ names}"
    by = {f["name"]: f for f in cfg["flags"]}
    assert by["size"]["default"] == pipeline.parse_args([]).size
    assert by["floor"]["default"] == pipeline.KEY_FLOOR
    assert by["place"]["choices"] == list(pipeline.PLACES)
    assert by["colors"]["default"] is None          # để trống nghĩa là không gom
    assert {f["group"] for f in cfg["flags"]} == {"chính", "nâng cao"}


def test_every_page_flag_is_accepted_by_the_runner(server, workspace):
    """Cờ nào trang hiện thì make_args phải nhận, nếu không bấm Chạy sẽ hỏng giữa chừng."""
    cfg = json.loads(get(server, "/api/config")[1])
    settings = {f["name"]: f["default"] for f in cfg["flags"]}
    args = ui.make_args(settings)
    for name, value in settings.items():
        assert getattr(args, name) == value


def test_audit_endpoint_grades_the_batch(server, workspace):
    src = workspace / "input" / "a.png"
    _design(src)
    pipeline.process_one(src, ui.make_args({"size": "300x300"}))
    rows = json.loads(get(server, "/api/audit")[1])
    assert [r["name"] for r in rows] == ["a_300x300_center.png"]
    assert rows[0]["dac"] > 80 and rows[0]["why"] == []

    bad = np.zeros((40, 40, 4), np.uint8)
    bad[..., 3] = 255
    Image.fromarray(bad, "RGBA").save(workspace / "output" / "a_300x300_top-left.png")
    rows = json.loads(get(server, "/api/audit")[1])
    flagged = [r for r in rows if r["why"]]
    assert len(flagged) == 1 and "fill-holes" in flagged[0]["why"][0]


def test_list_images_lists_every_print_file_of_an_image(workspace):
    """Một ảnh gốc có thể có nhiều file in (khác khung, vị trí, mực); trang phải thấy hết chứ
    không chỉ bản mới nhất."""
    _design(workspace / "input" / "a.png")
    out = workspace / "output"
    import os, time
    for i, n in enumerate(("a_4500x5100_center.png", "a_4500x5100_top-right_26pc.png")):
        Image.new("RGBA", (4, 4)).save(out / n)
        os.utime(out / n, (time.time() + i, time.time() + i))
    row = {r["name"]: r for r in ui.list_images()}["a.png"]
    assert row["outs"] == ["a_4500x5100_top-right_26pc.png", "a_4500x5100_center.png"]
    assert row["out"] == row["outs"][0]


def test_server_serves_the_print_file_asked_for(server, workspace):
    src = workspace / "input" / "a.png"
    _design(src)
    pipeline.process_one(src, ui.make_args({"size": "300x300"}))
    pipeline.process_one(src, ui.make_args({"size": "300x300", "place": "top-right", "scale": 26.0}))
    other = urllib.parse.quote("a_300x300_top-right_26pc.png")

    with urllib.request.urlopen(server + "/file/a.png?out=" + other) as r:  # noqa: S310
        assert "a_300x300_top-right_26pc.png" in r.headers["Content-Disposition"]
    with urllib.request.urlopen(server + "/file/a.png") as r:  # noqa: S310
        assert "a_300x300_top-right_26pc.png" in r.headers["Content-Disposition"], "không hỏi thì bản mới nhất"

    rep = json.loads(get(server, "/api/report/a.png?out=" + urllib.parse.quote("a_300x300_center.png"))[1])
    assert rep["out"] == "a_300x300_center.png"
    assert rep["meta"]["cmd"] == "./run.sh --size 300x300"      # trang hiện được cờ đã dùng
    rep = json.loads(get(server, "/api/report/a.png?out=" + other)[1])
    assert "--place top-right" in rep["meta"]["cmd"]

    status, body = get(server, "/out/a.png?out=" + other)
    assert status == 200 and Image.open(io.BytesIO(body)).size[0] <= ui.PREVIEW_PX

    # file của ảnh khác, hay tên bịa, không được trả về dưới tên ảnh này
    _design(workspace / "input" / "b.png")
    pipeline.process_one(workspace / "input" / "b.png", ui.make_args({"size": "300x300"}))
    for bad in ("b_300x300_center.png", "../input/a.png", "khong-co.png"):
        with pytest.raises(urllib.error.HTTPError) as e:
            get(server, "/file/a.png?out=" + urllib.parse.quote(bad))
        assert e.value.code == 404


def test_page_offers_the_presets_and_applies_them(workspace):
    flags = {f["name"]: f for f in ui.page_config()["flags"]}
    assert flags["preset"]["choices"] == ["none", *pipeline.PRESETS]
    src = workspace / "input" / "a.png"
    _design(src)
    pipeline.process_one(src, ui.make_args({"size": "300x300", "preset": "chest-left"}))
    assert (workspace / "output" / "a_300x300_top-right_26pc.png").exists()


def test_page_config_carries_what_each_preset_fills_in():
    """Trang điền Đặt và Cỡ % ngay khi chọn mẫu, nên phải biết mỗi mẫu điền gì, đọc từ chính PRESETS."""
    presets = ui.page_config()["presets"]
    assert presets == {k: {"place": p, "scale": s} for k, (p, s) in pipeline.PRESETS.items()}
    assert presets["chest-left"] == {"place": "top-right", "scale": 26.0}


def test_server_serves_a_soft_proof_of_the_print_file(server, workspace):
    if pipeline.CMYK_PROFILE is None:
        pytest.skip("không có hồ sơ CMYK trên máy này")
    src = workspace / "input" / "a.png"
    _design(src)
    pipeline.process_one(src, ui.make_args({"size": "300x300"}))
    status, body = get(server, "/proof/a.png")
    assert status == 200 and Image.open(io.BytesIO(body)).mode == "RGBA"
    rep = json.loads(get(server, "/api/report/a.png")[1])
    assert "gamut" in rep


def test_report_carries_the_run_notes_for_the_page(server, workspace, monkeypatch):
    monkeypatch.setattr(pipeline, "remove_bg", lambda img: Image.new("RGBA", img.size, (255, 255, 255, 255)))
    src = workspace / "input" / "a.png"
    _design(src)
    pipeline.process_one(src, ui.make_args({"size": "1500x1500", "fill_holes": True}))
    rep = json.loads(get(server, "/api/report/a.png")[1])
    assert isinstance(rep["meta"]["notes"], list) and rep["meta"]["notes"]
    assert any("ảnh gốc nhỏ" in n for n in rep["meta"]["notes"])
    assert rep["px"] == [1500, 1500]   # kích thước thật, kể cả với file tên kiểu cũ không mang khung in


def test_runner_status_tells_the_page_what_is_queued_running_and_how_long(workspace, monkeypatch):
    """Sidebar cần biết ảnh nào đang xếp hàng, ảnh nào đang chạy và đã chạy bao lâu, ảnh xong mất
    bao lâu: không có những thứ đó thì người dùng chỉ thấy một nút bị xám."""
    import time
    gate = threading.Event()
    out = workspace / "output" / "a_1x1_center.png"
    out.parent.mkdir(exist_ok=True)
    pipeline.save_print_png(Image.new("RGBA", (4, 4), (255, 0, 0, 255)), out, meta={"notes": ["GỢI Ý x"]})
    monkeypatch.setattr(pipeline, "process_one", lambda src, args: (gate.wait(2), out)[1])
    for n in ("a.png", "b.png"):
        _design(workspace / "input" / n)
    r = ui.Runner()
    r.start([("a.png", {}), ("b.png", {})], workers=1)
    time.sleep(0.15)
    st = r.status()
    assert st["total"] == 2 and st["queue"] == ["b.png"] and st["current"] == ["a.png"]
    assert 0 < st["elapsed"]["a.png"] < 2
    gate.set()
    for _ in range(100):
        if not r.status()["running"]:
            break
        time.sleep(0.05)
    st = r.status()
    assert set(st["took"]) == {"a.png", "b.png"} and st["queue"] == [] and st["elapsed"] == {}
    assert st["notes"]["a.png"] == ["GỢI Ý x"]   # trang bật toast từ đây, không cần đo lại file in


def test_a_failed_run_leaves_the_image_waiting(workspace, monkeypatch):
    """Ảnh lỗi phải còn trong hàng chờ để sửa cờ rồi chạy lại, không lẳng lặng sang done/."""
    import time
    def boom(src, args):
        raise RuntimeError("hỏng")
    monkeypatch.setattr(pipeline, "process_one", boom)
    _design(workspace / "input" / "a.png")
    r = ui.Runner()
    r.start([("a.png", {})], workers=1)
    for _ in range(100):
        if not r.status()["running"]:
            break
        time.sleep(0.02)
    assert "a.png" in r.status()["errors"] and (workspace / "input" / "a.png").exists()


def test_runner_takes_more_jobs_while_running(workspace, monkeypatch):
    """Bấm Chạy cho ảnh khác lúc đang chạy thì ảnh đó vào cuối hàng, không bị bỏ qua. Ảnh đã có trong
    hàng hoặc đang chạy thì không thêm lần nữa."""
    import time
    gate = threading.Event()
    monkeypatch.setattr(pipeline, "process_one", lambda src, args: gate.wait(3) and None)
    for n in ("a.png", "b.png", "c.png"):
        _design(workspace / "input" / n)
    r = ui.Runner()
    r.start([("a.png", {})], workers=1)
    time.sleep(0.1)
    r.start([("b.png", {}), ("a.png", {})], workers=1)
    r.start([("c.png", {}), ("b.png", {})], workers=1)
    st = r.status()
    assert st["current"] == ["a.png"] and st["queue"] == ["b.png", "c.png"] and st["total"] == 3
    gate.set()
    for _ in range(200):
        if not r.status()["running"]:
            break
        time.sleep(0.02)
    assert sorted(r.status()["done"]) == ["a.png", "b.png", "c.png"]


def test_outputs_for_finds_a_print_file_named_the_old_way(workspace):
    """File in làm trước khi tên mang khung và vị trí chỉ là <tên>.png. Trang vẫn phải mở được nó."""
    out = workspace / "output"
    Image.new("RGBA", (8, 8)).save(out / "poster.png")
    Image.new("RGBA", (8, 8)).save(out / "poster_4500x5400_center.png")
    Image.new("RGBA", (8, 8)).save(out / "posterx.png")          # ảnh khác, không được lẫn vào
    assert sorted(p.name for p in ui.outputs_for("poster")) == ["poster.png", "poster_4500x5400_center.png"]


def test_measure_is_cached_until_the_print_file_changes(workspace, monkeypatch):
    """Mở một ảnh không được đo lại file in 20 MB mỗi lần: đo một lần, dùng lại tới khi file đổi."""
    import os, time
    calls = []
    monkeypatch.setattr(pipeline, "measure_print", lambda out, src: calls.append(out) or {"dac": len(calls)})
    src, out = workspace / "input" / "a.png", workspace / "output" / "a_8x8_center.png"
    _design(src)
    Image.new("RGBA", (8, 8)).save(out)
    assert ui.measure(out, src) == {"dac": 1}
    assert ui.measure(out, src) == {"dac": 1} and len(calls) == 1
    os.utime(out, (time.time() + 5, time.time() + 5))
    assert ui.measure(out, src) == {"dac": 2}


def test_page_config_names_every_choice_in_vietnamese():
    """Lựa chọn hiện tiếng Việt; giá trị gửi đi vẫn là giá trị của dòng lệnh."""
    flags = {f["name"]: f for f in ui.page_config()["flags"]}
    for f in flags.values():
        if f["choices"]:
            assert set(f["labels"]) == set(f["choices"]), f["name"]
    assert flags["place"]["labels"]["top-right"] == "góc trên phải"


def test_delete_removes_an_image_and_everything_made_from_it(server, workspace):
    """Xóa ảnh: ảnh gốc, mọi file in, ảnh so sánh, file tạm. Không đụng ảnh khác tên gần giống."""
    _design(workspace / "input" / "done" / "a.png")
    _design(workspace / "input" / "ab.png")
    for d, n in (("output", "a_8x8_center.png"), ("output", "a.png"), ("review", "a_8x8_center.png"),
                 ("work", "a-cut.png"), ("output", "ab_8x8_center.png")):
        Image.new("RGBA", (8, 8)).save(workspace / d / n)
    req = urllib.request.Request(server + "/api/delete?name=a.png", b"", OK, method="POST")
    with urllib.request.urlopen(req) as r:  # noqa: S310
        assert json.loads(r.read())["removed"] == 5
    left = sorted(p.relative_to(workspace).as_posix() for p in workspace.rglob("*.png"))
    assert left == ["input/ab.png", "output/ab_8x8_center.png"]


def test_delete_one_print_file_keeps_the_image(server, workspace):
    _design(workspace / "input" / "a.png")
    for n in ("a_8x8_center.png", "a_9x9_center.png"):
        Image.new("RGBA", (8, 8)).save(workspace / "output" / n)
        Image.new("RGBA", (8, 8)).save(workspace / "review" / n)
    req = urllib.request.Request(server + "/api/delete?name=a.png&out=a_8x8_center.png", b"", OK, method="POST")
    urllib.request.urlopen(req)  # noqa: S310
    assert [p.name for p in ui.outputs_for("a")] == ["a_9x9_center.png"]
    assert not (workspace / "review" / "a_8x8_center.png").exists() and (workspace / "input" / "a.png").exists()


def test_delete_refuses_an_image_that_is_running(server, workspace, monkeypatch):
    _design(workspace / "input" / "a.png")
    monkeypatch.setattr(ui.RUNNER, "busy", {"a.png"})
    req = urllib.request.Request(server + "/api/delete?name=a.png", b"", OK, method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)  # noqa: S310
    assert e.value.code == 400 and (workspace / "input" / "a.png").exists()


def test_page_is_read_again_when_ui_html_changes(server, workspace, tmp_path, monkeypatch):
    """Sửa ui.html không cần khởi động lại server: trang đọc lại file khi nó đổi."""
    import os, time
    page = tmp_path / "ui.html"
    page.write_text("<title>v1</title>", encoding="utf-8")
    monkeypatch.setattr(ui, "PAGE_FILE", page)
    assert b"v1" in get(server, "/")[1]
    page.write_text("<title>v2</title>", encoding="utf-8")
    os.utime(page, (time.time() + 5, time.time() + 5))
    assert b"v2" in get(server, "/")[1]


def _status(req):
    try:
        with urllib.request.urlopen(req) as r:  # noqa: S310
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def test_post_without_the_page_header_is_refused(server, workspace):
    """Chặn CSRF: một trang web lạ mở trong cùng trình duyệt gửi được POST "đơn giản" tới 127.0.0.1,
    nhưng không gửi được header riêng mà không qua CORS preflight, thứ server này không bao giờ cho."""
    _design(workspace / "input" / "a.png")
    for path in ("/api/delete?name=a.png", "/api/run", "/api/upload?name=b.png"):
        assert _status(urllib.request.Request(server + path, b"{}", method="POST")) == 403, path
    assert (workspace / "input" / "a.png").exists()


def test_a_request_from_another_site_is_refused(server, workspace):
    """Origin lạ hoặc Host lạ (DNS rebinding: tên miền lạ trỏ về 127.0.0.1) đều bị từ chối, kể cả GET."""
    _design(workspace / "input" / "a.png")
    evil = urllib.request.Request(server + "/api/delete?name=a.png", b"", {**OK, "Origin": "https://evil.example"}, method="POST")
    assert _status(evil) == 403 and (workspace / "input" / "a.png").exists()
    port = server.rsplit(":", 1)[1]
    assert _status(urllib.request.Request(server + "/api/images", headers={"Host": f"evil.example:{port}"})) == 403
    assert _status(urllib.request.Request(server + "/api/images", headers={"Host": f"localhost:{port}"})) == 200
    same = urllib.request.Request(server + "/api/delete?name=a.png", b"", {**OK, "Origin": server}, method="POST")
    assert _status(same) == 200


def test_names_cannot_climb_out_of_input(server, workspace):
    """Tên ảnh từ trình duyệt không được trỏ ra ngoài input/: không đọc, không xóa được file nào khác."""
    outside = workspace / "bimat.png"
    _design(outside)
    _design(workspace / "input" / "a.png")
    for bad in ("../bimat.png", "done/../../bimat.png", "..", "/etc/hosts", "a.png/", ".hidden.png"):
        with pytest.raises(FileNotFoundError):
            ui.source_path(bad)
        q = urllib.parse.quote(bad, safe="")
        assert _status(urllib.request.Request(server + "/api/delete?name=" + q, b"", OK, method="POST")) == 400, bad
    assert _status(urllib.request.Request(server + "/src/" + urllib.parse.quote("../bimat.png", safe=""))) == 404
    assert outside.exists()
