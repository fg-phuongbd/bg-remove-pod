"""Trang xem và chạy tại máy: mở bằng `./run.sh --ui`.

Ba việc mà dòng lệnh làm không tốt, và đây là lý do trang này tồn tại:

* Xem kết quả trên **đúng màu áo sẽ in**. File cho áo tối phần lớn là mực sáng, nên mở trên nền
  trắng hay checkerboard sáng thì nét trắng biến mất và trông như thủng trong khi không hề thủng.
* Chọn cờ **theo từng ảnh**, thay vì một bộ cờ cho cả lô rồi phải chạy lại riêng vài ảnh.
* Hiện các con số chấm chất lượng ngay cạnh ảnh, thay vì phải đo thủ công.

Phần tính toán nằm ở các hàm thuần phía trên; phần HTTP chỉ bọc quanh chúng.
"""
from __future__ import annotations

import argparse
import errno
import glob
import json
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image

import pipeline

PREVIEW_PX = 1400  # ảnh xem trên trang, đủ nét để soi mép mà không phải tải file in 20 MB
THUMB_PX = 420
CACHE = pipeline.WORK_DIR / "ui"


# ---------------------------------------------------------------- đo chất lượng
measure = pipeline.measure_print  # cùng một phép đo với lệnh ./run.sh --audit


# ---------------------------------------------------------------- cờ cho từng ảnh
def make_args(settings: dict) -> argparse.Namespace:
    """Bộ cờ của một ảnh, dựng từ mặc định của dòng lệnh rồi ghi đè.

    Lấy mặc định bằng chính parse_args để trang không bao giờ lệch khỏi CLI khi có cờ mới.
    Luôn giữ ảnh gốc tại chỗ: trên trang, chạy lại cùng một ảnh với cờ khác là việc thường xuyên.
    """
    args = pipeline.parse_args([])
    args.keep_input = True
    for k, v in settings.items():
        if not hasattr(args, k):
            raise ValueError(f"không có cờ {k!r}")
        setattr(args, k, v)
    return args


# Cờ trang cho chỉnh, kèm nhóm hiện ở đâu. Giá trị mặc định và danh sách lựa chọn đều lấy từ
# parse_args chứ không chép lại, để trang không bao giờ lệch khỏi dòng lệnh khi CLI đổi.
PAGE_FLAGS = {
    "size":       ("chính", "khung in", ""),
    "shirt":      ("chính", "áo", "in lên áo cùng màu nền ảnh hay khác màu"),
    "fill_holes": ("chính", "thân hình đặc", "ảnh có người: tô đặc thân hình thay vì đục lỗ"),
    "preset":     ("chính", "mẫu vị trí", "vị trí in hay dùng, điền sẵn đặt và cỡ; chest-left = ngực trái người mặc"),
    "place":      ("chính", "đặt", "vị trí thiết kế trên khung in"),
    "scale":      ("chính", "cỡ %", "thiết kế chiếm bao nhiêu phần trăm khung in"),
    "margin":     ("chính", "lề %", "khoảng hở từ mép khung khi đặt lệch tâm"),
    "ink":        ("chính", "mực", "in một màu duy nhất, alpha thành độ phủ"),
    "vector":     ("chính", "vector", "gom màu rồi trace vector"),
    "style":      ("nâng cao", "kiểu", "ép model upscale"),
    "colors":     ("nâng cao", "gom màu", "gom về tối đa N màu, để trống là không gom"),
    "merge":      ("nâng cao", "ngưỡng gộp", "gộp hai màu gần nhau, CIELAB ΔE"),
    "floor":      ("nâng cao", "sàn màu nền", "màu cách nền dưới ngưỡng này cho trong suốt hẳn"),
    "fill_limit": ("nâng cao", "chặn tô đặc", "bỏ qua tô đặc nếu thêm quá N% mực in đè lên áo"),
    "bg":         ("nâng cao", "ép cách xử lý", "ép thẳng khi nhận diện nền sai"),
    "dtf_safe":   ("nâng cao", "nới nét DTF", "nới nét và đốm mảnh hơn 0,5 mm ra 0,5 mm để không bong"),
    "no_clean":   ("nâng cao", "giữ mực vô hình", "không dọn alpha dưới 8 và các đốm nhỏ mà mờ"),
    "dtf_warn":   ("nâng cao", "cảnh báo DTF %", "báo khi quá N% diện tích mực dưới 40% độ phủ"),
}


def page_config() -> dict:
    """Mặc định và lựa chọn cho từng cờ, đọc thẳng từ bộ phân tích tham số của dòng lệnh."""
    defaults = vars(pipeline.parse_args([]))
    choices = {
        "shirt": ["same", "other", "auto"],
        "place": list(pipeline.PLACES),
        "preset": ["none", *pipeline.PRESETS],
        "bg": ["auto", *pipeline.BG_KINDS, "ai"],
        "style": ["auto", "flat", "detail", "grain"],
        "ink": ["none", "black", "white"],
    }
    return {"flags": [{"name": k, "group": g, "label": label, "hint": hint,
                       "default": defaults[k], "choices": choices.get(k),
                       "kind": type(defaults[k]).__name__}
                      for k, (g, label, hint) in PAGE_FLAGS.items()]}


def list_images() -> list[dict]:
    """Ảnh đang chờ trong input/ và ảnh đã xử lý trong input/done/, mới nhất lên trước."""
    seen: dict[str, dict] = {}
    for folder, waiting in ((pipeline.INPUT_DIR, True), (pipeline.INPUT_DIR / "done", False)):
        if not folder.is_dir():
            continue
        for p in folder.iterdir():
            if not p.is_file() or p.suffix.lower() not in pipeline.IMAGE_EXTS or p.name in seen:
                continue
            outs = outputs_for(p.stem)
            seen[p.name] = {"name": p.name, "waiting": waiting, "mtime": p.stat().st_mtime,
                            "done": bool(outs), "out": outs[0].name if outs else None,
                            "outs": [o.name for o in outs]}
    return sorted(seen.values(), key=lambda r: -r["mtime"])


def outputs_for(stem: str) -> list[Path]:
    """File in của một ảnh gốc, mới nhất trước.

    Một ảnh gốc giờ có thể sinh nhiều file in, vì tên mang theo khung và vị trí: chạy lại cùng
    ảnh ở cỡ khác sẽ ra file khác chứ không đè lên nhau."""
    if not pipeline.OUTPUT_DIR.is_dir():
        return []
    hits = [p for p in pipeline.OUTPUT_DIR.glob(f"{glob.escape(stem)}_*.png")]
    return sorted(hits, key=lambda p: -p.stat().st_mtime)


def pick_output(name: str, wanted: str | None) -> Path:
    """File in của ảnh gốc `name`: bản có tên `wanted` nếu hỏi, không thì bản mới nhất.

    Tên hỏi phải là một trong các file in của đúng ảnh này, không được là file của ảnh khác hay
    một đường dẫn bịa: trang chỉ chọn trong danh sách nó đã được cho xem."""
    outs = outputs_for(source_path(name).stem)
    if not outs:
        raise FileNotFoundError(name)
    if not wanted:
        return outs[0]
    for o in outs:
        if o.name == wanted:
            return o
    raise FileNotFoundError(f"{name} không có file in {wanted!r}")


def source_path(name: str) -> Path:
    """Ảnh gốc theo tên, dù nó còn ở input/ hay đã sang input/done/."""
    for folder in (pipeline.INPUT_DIR, pipeline.INPUT_DIR / "done"):
        p = folder / name
        if p.is_file():
            return p
    raise FileNotFoundError(name)


def preview(path: Path, box: int, tag: str) -> Path:
    """Bản thu nhỏ để hiện trên trang, dựng một lần rồi dùng lại cho tới khi file gốc đổi."""
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / f"{tag}-{box}-{path.stem}.png"
    if dest.exists() and dest.stat().st_mtime >= path.stat().st_mtime:
        return dest
    im = Image.open(path).convert("RGBA")
    im.thumbnail((box, box), Image.Resampling.LANCZOS)
    im.save(dest, "PNG")
    return dest


# ---------------------------------------------------------------- hàng chạy
class Runner:
    """Chạy nhiều ảnh cùng lúc trong luồng nền, trang hỏi tiến độ bằng cách gọi lại.

    Phần lớn thời gian một ảnh nằm ở tiến trình con Real-ESRGAN, nên chạy song song vài ảnh
    rút ngắn được đáng kể. Mặc định 2 vì mỗi ảnh giữ vài mảng cỡ 5000 x 5000 trong bộ nhớ."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.queue: list[tuple[str, dict]] = []
        self.busy: set[str] = set()
        self.done: list[str] = []
        self.errors: dict[str, str] = {}
        self.workers = 0

    def status(self) -> dict:
        with self.lock:
            return {"running": self.workers > 0, "current": sorted(self.busy),
                    "done": list(self.done), "errors": dict(self.errors), "left": len(self.queue)}

    def start(self, jobs: list[tuple[str, dict]], workers: int = 2) -> None:
        with self.lock:
            if self.workers:
                return
            self.reset()
            self.queue = list(jobs)
            self.workers = max(1, min(int(workers), len(jobs) or 1))
            count = self.workers
        for _ in range(count):
            threading.Thread(target=self._work, daemon=True).start()

    def _work(self) -> None:
        while True:
            with self.lock:
                if not self.queue:
                    self.workers -= 1
                    return
                name, settings = self.queue.pop(0)
                self.busy.add(name)
            try:
                pipeline.process_one(source_path(name), make_args(settings))
                with self.lock:
                    self.done.append(name)
            except Exception:  # noqa: BLE001 - một ảnh hỏng không được làm chết cả hàng
                with self.lock:
                    self.errors[name] = traceback.format_exc()  # đủ traceback: trang hiện được cả
            finally:
                with self.lock:
                    self.busy.discard(name)


RUNNER = Runner()


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args) -> None:  # trang chạy tại máy, không cần log từng request
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: object, code: int = 200) -> None:
        self._send(code, json.dumps(data).encode(), "application/json; charset=utf-8")

    def _file(self, path: Path) -> None:
        self._send(200, path.read_bytes(), "image/png")

    def do_GET(self) -> None:  # noqa: N802 - tên do BaseHTTPRequestHandler quy định
        route = urlparse(self.path)
        parts = [unquote(s) for s in route.path.strip("/").split("/")]
        wanted = parse_qs(route.query).get("out", [None])[0]  # file in nào của ảnh, nếu hỏi
        try:
            if route.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            elif route.path in ("/", "/index.html"):
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif route.path == "/api/images":
                self._json(list_images())
            elif route.path == "/api/status":
                self._json(RUNNER.status())
            elif route.path == "/api/config":
                self._json(page_config())
            elif route.path == "/api/audit":
                srcs = [source_path(r["name"]) for r in list_images()]
                self._json(pipeline.audit_rows(srcs))
            elif parts[0] == "api" and parts[1] == "report" and len(parts) == 3:
                src = source_path(parts[2])
                if not outputs_for(src.stem):
                    self._json({})
                else:
                    out = pick_output(parts[2], wanted)
                    rep = measure(out, src)
                    self._json(dict(rep, **pipeline.print_verdict(rep), out=out.name,
                                    meta=pipeline.read_meta(out)))
            elif parts[0] == "src" and len(parts) == 2:
                self._file(preview(source_path(parts[1]), THUMB_PX, "src"))
            elif parts[0] == "out" and len(parts) == 2:
                self._file(preview(pick_output(parts[1], wanted), PREVIEW_PX, "out"))
            elif parts[0] == "file" and len(parts) == 2:
                out = pick_output(parts[1], wanted)
                body = out.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Content-Disposition", f'attachment; filename="{out.name}"')
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json({"error": "không có đường dẫn này"}, 404)
        except FileNotFoundError as e:
            self._json({"error": str(e)}, 404)
        except Exception as e:  # noqa: BLE001 - trả lỗi ra trang thay vì chết server
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def _upload(self, route) -> None:
        """Ghi một file kéo thả vào input/. Thân request là dữ liệu thô, tên nằm ở query."""
        try:
            name = safe_name(unquote(parse_qs(route.query).get("name", [""])[0]))
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= MAX_UPLOAD:
                raise ValueError(f"dung lượng không hợp lệ: {length} byte")
            pipeline.INPUT_DIR.mkdir(parents=True, exist_ok=True)
            dest = pipeline.INPUT_DIR / name
            dest.write_bytes(self.rfile.read(length))
            Image.open(dest).verify()  # từ chối file không phải ảnh thay vì để pipeline chết sau
            self._json({"name": name})
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"{type(e).__name__}: {e}"}, 400)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path)
        if route.path == "/api/upload":
            self._upload(route)
            return
        if route.path != "/api/run":
            self._json({"error": "không có đường dẫn này"}, 404)
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            jobs = [(j["name"], j.get("settings", {})) for j in body.get("jobs", [])]
            for _, settings in jobs:
                make_args(settings)  # kiểm cờ trước khi nhận, để lỗi hiện ngay chứ không giữa chừng
            RUNNER.start(jobs, int(body.get("workers", 2)))
            self._json(RUNNER.status())
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"{type(e).__name__}: {e}"}, 400)


MAX_UPLOAD = 64 * 1024 * 1024


def safe_name(raw: str) -> str:
    """Tên file do trình duyệt gửi lên, gọt cho an toàn để ghi vào input/.

    Chỉ giữ phần tên cuối và bắt buộc phải là đuôi ảnh: tên từ bên ngoài không được phép trỏ ra
    ngoài thư mục input/."""
    name = Path(raw.replace("\\", "/")).name
    if not name or name.startswith(".") or Path(name).suffix.lower() not in pipeline.IMAGE_EXTS:
        raise ValueError(f"tên file không nhận: {raw!r}")
    return name


def serve(port: int = 8765, open_browser: bool = True) -> None:
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
        print(f"Cổng {port} đang bận: nhiều khả năng một trang khác đang mở ở http://127.0.0.1:{port}/\n"
              f"Mở lại tab đó, hoặc tắt tiến trình cũ rồi chạy lại. Sau khi sửa code cũng phải tắt và "
              f"chạy lại thì trang mới cập nhật.")
        return
    url = f"http://127.0.0.1:{port}/"
    print(f"Trang xem đang chạy tại {url}  (Ctrl+C để dừng)")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng.")
    finally:
        server.server_close()


PAGE = (Path(__file__).parent / "ui.html").read_text(encoding="utf-8")
