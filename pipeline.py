"""tshirt-pipeline: AI flat illustration -> print-ready transparent PNG (300 DPI).

Default mode traces the design to vector (vtracer) and re-renders it (resvg).
--raster mode upscales (Real-ESRGAN, Lanczos fallback) and flattens colors instead.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ---------------------------------------------------------------- constants
ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
REVIEW_DIR = ROOT / "review"
WORK_DIR = ROOT / "work"
BIN_DIR = ROOT / "bin"
REALESRGAN_BIN = BIN_DIR / "realesrgan-ncnn-vulkan"
DPI = 300
REMBG_MODEL = "birefnet-general"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
VTRACER_OPTS = dict(
    colormode="color", hierarchical="stacked", mode="spline",
    filter_speckle=4, color_precision=8, corner_threshold=60, path_precision=3,
)


class EmptyResult(Exception):
    """Background removal left no visible pixels."""


# ---------------------------------------------------------------- geometry
def target_box_px(size: str) -> tuple[int, int]:
    """'30x40' (cm) -> (3543, 4724) px at 300 DPI."""
    w_cm, h_cm = (float(v) for v in size.lower().split("x"))
    return round(w_cm / 2.54 * DPI), round(h_cm / 2.54 * DPI)


def fit_box(w: int, h: int, box: tuple[int, int]) -> tuple[int, int]:
    """Largest (w, h) with the same aspect ratio that fits inside box."""
    scale = min(box[0] / w, box[1] / h)
    return max(1, int(w * scale)), max(1, int(h * scale))  # floor: never exceed the box


def crop_to_content(img: Image.Image, margin: float = 0.02) -> Image.Image:
    """Crop to the bounding box of alpha > 0, padded by margin per side."""
    alpha = np.asarray(img.convert("RGBA"))[:, :, 3]
    ys, xs = np.nonzero(alpha)
    if len(xs) == 0:
        raise EmptyResult("tách nền ra rỗng, không còn pixel nào")
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad_x, pad_y = round((x1 - x0) * margin), round((y1 - y0) * margin)
    out = Image.new("RGBA", (x1 - x0 + 2 * pad_x, y1 - y0 + 2 * pad_y), (0, 0, 0, 0))
    out.paste(img.convert("RGBA").crop((x0, y0, x1, y1)), (pad_x, pad_y))
    return out


# ---------------------------------------------------------------- color
def quantize(img: Image.Image, colors: int, binary_alpha: bool) -> Image.Image:
    """Reduce RGB to `colors` flat colors; keep alpha separately."""
    rgba = np.asarray(img.convert("RGBA")).copy()
    alpha = rgba[:, :, 3]
    opaque = alpha > 0
    rgb = rgba[:, :, :3]
    if opaque.any():
        # fill transparent pixels with the mean opaque color so they do not steal palette slots
        rgb[~opaque] = rgb[opaque].mean(axis=0).astype(np.uint8)
    q = Image.fromarray(rgb, "RGB").quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    flat = np.asarray(q.convert("RGB"))
    if binary_alpha:
        alpha = np.where(alpha >= 128, 255, 0).astype(np.uint8)
    else:
        alpha = np.where(alpha < 16, 0, np.where(alpha > 240, 255, alpha)).astype(np.uint8)
    return Image.fromarray(np.dstack([flat, alpha]), "RGBA")


# ---------------------------------------------------------------- vector
def check_tools() -> None:
    if shutil.which("resvg") is None:
        sys.exit("Thiếu resvg. Chạy ./setup.sh trước.")


def trace_svg(png_path: Path, svg_path: Path) -> None:
    import vtracer  # heavy import kept local

    vtracer.convert_image_to_svg_py(str(png_path), str(svg_path), **VTRACER_OPTS)


def _svg_size(svg_path: Path) -> tuple[int, int]:
    head = svg_path.read_text(errors="ignore")[:2000]
    w = re.search(r'<svg[^>]*\swidth="([\d.]+)', head)
    h = re.search(r'<svg[^>]*\sheight="([\d.]+)', head)
    if not (w and h):
        raise ValueError("SVG không có width/height")
    return round(float(w.group(1))), round(float(h.group(1)))


def render_svg(svg_path: Path, box: tuple[int, int]) -> Image.Image:
    """Render SVG with resvg so that it fits inside box; transparent background."""
    w, h = fit_box(*_svg_size(svg_path), box)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.png"
        subprocess.run(["resvg", "-w", str(w), "-h", str(h), str(svg_path), str(out)],
                       check=True, capture_output=True, text=True)
        return Image.open(out).convert("RGBA").copy()


# ---------------------------------------------------------------- raster
def upscale(img: Image.Image, scale: int = 4) -> Image.Image:
    """Real-ESRGAN anime model if the binary exists, else Lanczos."""
    img = img.convert("RGBA")
    if REALESRGAN_BIN.exists():
        with tempfile.TemporaryDirectory() as td:
            src, dst = Path(td) / "in.png", Path(td) / "out.png"
            img.save(src)
            subprocess.run([str(REALESRGAN_BIN), "-i", str(src), "-o", str(dst),
                            "-n", "realesrgan-x4plus-anime", "-s", str(scale),
                            "-m", str(BIN_DIR / "models")],
                           check=True, capture_output=True, text=True)
            return Image.open(dst).convert("RGBA").copy()
    print("  CẢNH BÁO: không có Real-ESRGAN trong bin/, dùng Lanczos thay thế")
    return img.resize((img.width * scale, img.height * scale), Image.Resampling.LANCZOS)


def flatten_raster(img: Image.Image, colors: int) -> Image.Image:
    """Median-filter noise away, then quantize while keeping soft alpha."""
    rgba = img.convert("RGBA")
    rgb = rgba.convert("RGB").filter(ImageFilter.MedianFilter(5))
    rgb.putalpha(rgba.getchannel("A"))
    return quantize(rgb, colors, binary_alpha=False)


# ---------------------------------------------------------------- background
_SESSION = None


def remove_bg(img: Image.Image) -> Image.Image:
    global _SESSION
    import rembg  # heavy import kept local

    if _SESSION is None:
        print(f"  Nạp model {REMBG_MODEL} (lần đầu sẽ tải ~900 MB về ~/.u2net/)...")
        _SESSION = rembg.new_session(REMBG_MODEL)
    return rembg.remove(img.convert("RGBA"), session=_SESSION, alpha_matting=False).convert("RGBA")


# ---------------------------------------------------------------- export
def save_print_png(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGBA").save(path, "PNG", dpi=(DPI, DPI), optimize=False)


def _checkerboard(size: tuple[int, int], cell: int = 32) -> Image.Image:
    w, h = size
    tile = np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.indices((h, w))
    tile[((yy // cell) + (xx // cell)) % 2 == 0] = 255
    tile[tile == 0] = 200
    return Image.fromarray(tile, "L").convert("RGBA")


def make_review(original: Image.Image, result: Image.Image, path: Path, height: int = 800, gap: int = 20) -> None:
    """Original on the left, result on a checkerboard on the right."""
    def scaled(im: Image.Image) -> Image.Image:
        return im.convert("RGBA").resize((max(1, round(im.width * height / im.height)), height), Image.Resampling.LANCZOS)

    left, right = scaled(original), scaled(result)
    canvas = Image.new("RGBA", (left.width + gap + right.width, height), (255, 255, 255, 255))
    canvas.paste(left, (0, 0), left)
    board = _checkerboard(right.size)
    board.alpha_composite(right)
    canvas.paste(board, (left.width + gap, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, "PNG")


# ---------------------------------------------------------------- pipeline
def process_one(src: Path, args: argparse.Namespace) -> Path:
    box = target_box_px(args.size)
    original = Image.open(src)
    original.load()
    original = original.convert("RGBA")

    cut = crop_to_content(remove_bg(original))
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    cut_path = WORK_DIR / f"{src.stem}-cut.png"
    cut.save(cut_path)

    if args.raster:
        big = upscale(cut)
        flat = flatten_raster(big, args.colors)
        result = flat.resize(fit_box(*flat.size, box), Image.Resampling.LANCZOS)
    else:
        q = quantize(cut, args.colors, binary_alpha=True)
        q_path = WORK_DIR / f"{src.stem}-quant.png"
        q.save(q_path)
        svg_path = WORK_DIR / f"{src.stem}.svg"
        trace_svg(q_path, svg_path)
        result = render_svg(svg_path, box)

    out_path = OUTPUT_DIR / f"{src.stem}.png"
    save_print_png(result, out_path)
    make_review(original, result, REVIEW_DIR / f"{src.stem}.png")
    return out_path


def _collect(args: argparse.Namespace) -> list[Path]:
    if args.files:
        return [Path(f) for f in args.files]
    return sorted(p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _move(src: Path, sub: str) -> None:
    dest = INPUT_DIR / sub
    dest.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest / src.name))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI flat illustration -> print-ready transparent PNG (300 DPI).")
    p.add_argument("--raster", action="store_true", help="dùng chế độ raster (upscale + gom màu) thay cho vector")
    p.add_argument("--colors", type=int, default=12, help="số màu tối đa sau khi gom (mặc định 12)")
    p.add_argument("--size", default="30x40", help="khung in theo cm, dạng WxH (mặc định 30x40)")
    p.add_argument("--keep-input", action="store_true", help="không chuyển ảnh gốc sang input/done/")
    p.add_argument("files", nargs="*", help="chỉ xử lý các file này thay cho cả input/")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    check_tools()
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = _collect(args)
    if not files:
        print(f"Không có ảnh nào trong {INPUT_DIR}")
        return 0
    mode = "raster" if args.raster else "vector"
    box = target_box_px(args.size)
    print(f"{len(files)} ảnh | chế độ {mode} | {args.colors} màu | khung {args.size} cm = {box[0]}x{box[1]} px")

    ok, failed = [], []
    for i, src in enumerate(files, 1):
        t0 = time.time()
        print(f"[{i}/{len(files)}] {src.name}")
        try:
            out = process_one(src, args)
            ok.append(src)
            print(f"  -> {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out} ({time.time() - t0:.1f}s)")
            if not args.keep_input and src.parent == INPUT_DIR:
                _move(src, "done")
        except Exception as e:  # noqa: BLE001 - one bad file must not stop the batch
            reason = str(e) or e.__class__.__name__
            failed.append((src, reason))
            print(f"  LỖI: {reason}")
            if src.parent == INPUT_DIR:
                _move(src, "failed")

    print(f"\nXong: {len(ok)} thành công, {len(failed)} lỗi.")
    for src, reason in failed:
        print(f"  {src.name}: {reason}")
    if ok:
        print(f"Kết quả trong {OUTPUT_DIR}, ảnh so sánh trong {REVIEW_DIR}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
