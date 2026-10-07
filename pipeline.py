"""Clearcut: ảnh thiết kế AI -> file in PNG nền trong suốt, 300 DPI.

Mặc định là raster: tách nền (key theo màu nền, hoặc cắt hình bằng model khi in lên áo khác
màu), upscale 4 lần bằng Real-ESRGAN (Lanczos khi không có), đặt lên khung in. `--vector` gom
màu rồi trace vector (vtracer) và vẽ lại (resvg) cho minh họa phẳng. Trang xem tại máy ở ui.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

# ---------------------------------------------------------------- constants
ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
REVIEW_DIR = ROOT / "review"
WORK_DIR = ROOT / "work"
BIN_DIR = ROOT / "bin"
REALESRGAN_BIN = BIN_DIR / "realesrgan-ncnn-vulkan"
DPI = 300
UPSCALE = 4  # Real-ESRGAN phóng 4 lần; xa hơn là Lanczos kéo giãn
MODEL_MIN_GROW = 2.0  # chỉ cần phóng tới mức này thì ảnh gốc đã đủ chi tiết, Lanczos là đủ
# Ảnh ra khỏi model tối đa bao nhiêu pixel. Các bước sau tốn khoảng 130 byte mỗi pixel: ảnh ChatGPT
# 1254 px ra 25 triệu pixel (~3 GB), còn ảnh 3840x2160 qua model 4 lần ra 132 triệu (17 GB, hết RAM).
MODEL_MAX_PX = 40_000_000
DEFAULT_SIZE = "4500x5400"  # px; print file 38.1 x 45.7 cm at 300 DPI
REMBG_MODEL = "birefnet-general"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
DTF_COVERAGE = 102  # dưới 40% độ phủ: vùng nhận ít bột keo khi in DTF, dễ bong sau vài lần giặt
DTF_WARN = 5.0  # cảnh báo khi vùng phủ thấp vượt quá ngần này phần trăm diện tích mực
THIN_WARN = 5.0  # cảnh báo khi quá ngần này phần trăm mực nằm trong nét mảnh hơn MIN_FEATURE_MM
SPECK_WARN = 1.0  # cảnh báo khi quá ngần này phần trăm mực là đốm rời nhỏ hơn SPECK_MM2
# --fill-holes bị bỏ qua nếu nó thêm quá ngần này phần trăm mực trùng màu áo. Mặc định 100 = tắt:
# người dùng chỉ bật cờ cho ảnh có người và muốn người là một khối đặc đúng như model cắt, kể cả
# quần đen trên áo đen (ảnh đen trắng thêm tới 33%). Ai lo bật nhầm cho poster thì đặt 20: ba ảnh
# người màu thêm 8-12%, ba poster thêm 27-60%.
FILL_LIMIT = 100.0
# Gợi ý bật thân hình đặc khi một ảnh chụp, key xong, thủng quá ngần này phần trăm thân hình. Đo trên
# 7 ảnh: cầu thủ trên nền đen thủng 16%, logo phẳng 1%; poster halftone thủng 35-56% nhưng đã bị
# loại từ trước vì kiểu "grain", và tô đặc poster là sai.
SEE_THROUGH_HINT = 8.0
SOLID_SHARE = 0.5  # share of same-colored neighbours that makes a pixel part of a flat area
KEY_FLOOR = 32.0  # default for --floor: colors closer than this to the background are shirt, not ink
MERGE_DELTA_E = 12.0  # default for --merge: palette colors closer than this (CIELAB) are always merged
VTRACER_OPTS = dict(
    colormode="color", hierarchical="stacked", mode="spline",
    color_precision=8, corner_threshold=60, path_precision=3,
)  # filter_speckle is computed per image in trace_svg


class EmptyResult(Exception):
    """Background removal left no visible pixels."""


# ---------------------------------------------------------------- geometry
def load_image(path: Path) -> Image.Image:
    """Mở ảnh gốc, xoay theo cờ EXIF, trả về RGBA.

    Máy ảnh và điện thoại hay lưu ảnh nằm ngang kèm một cờ bảo phần mềm xoay lại khi hiển thị.
    Đọc thô thì được đúng dữ liệu nhưng sai hướng, và cả file in lẫn màu nền đo được đều lệch theo.
    Mọi chỗ đọc ảnh gốc đều phải đi qua đây."""
    im = Image.open(path)
    im.load()
    return ImageOps.exif_transpose(im).convert("RGBA")


def out_name(stem: str, box: tuple[int, int], place: str, scale: float, ink: str = "none",
             dtf_safe: bool = False, *, fill: bool = False, cutout: bool = False, vector: bool = False,
             style: str = "auto", colors: int | None = None, halftone: bool = False) -> str:
    """Tên file in: tên ảnh, khung in, vị trí, rồi một đuôi cho mỗi cờ khác mặc định làm đổi bức ảnh.

    Mặc định vẫn gọn (`name_4500x5100_center.png`). Mọi cờ đổi kết quả đều để dấu trong tên,
    theo thứ tự cố định, để hai lần chạy khác cờ ra hai file thay vì lần sau đè lần trước: cỡ,
    màu mực, cắt hình (áo khác màu), thân hình đặc, vector, kiểu upscale ép tay, gom màu, nới nét,
    chấm hóa vùng mờ.
    `fill` là đã tô đặc thật, không phải cờ đã bật: chốt chặn bỏ qua thì file không mang đuôi."""
    tail = "" if scale == 100.0 else f"_{scale:g}pc"
    tail += "" if ink.strip().lower() in ("", "none") else f"_ink-{ink.strip().lower().lstrip('#')}"
    tail += "_cutout" if cutout else ""
    tail += "_fill" if fill else ""
    tail += "_vector" if vector else ""
    tail += f"_{style}" if style not in ("auto", "", None) else ""
    tail += f"_c{colors}" if colors else ""
    tail += "_dtf-safe" if dtf_safe else ""
    tail += "_ht" if halftone else ""
    return f"{stem}_{box[0]}x{box[1]}_{place}{tail}.png"


def target_box_px(size: str) -> tuple[int, int]:
    """'4500x5100' -> pixels as given; '30x40' (values < 200 are cm) -> (3543, 4724) at 300 DPI."""
    def num(tok: str) -> float:
        tok = tok.strip().replace(",", "")
        if re.fullmatch(r"\d{1,3}\.\d{3}", tok):  # Vietnamese thousands dot: 4.500 -> 4500
            tok = tok.replace(".", "")
        return float(tok)

    w, h = (num(v) for v in size.lower().split("x"))
    if w >= 200 and h >= 200:
        return round(w), round(h)
    return round(w / 2.54 * DPI), round(h / 2.54 * DPI)


PLACES = ("center", "top", "bottom", "left", "right",
          "top-left", "top-right", "bottom-left", "bottom-right")

# Vị trí in hay dùng, tên theo người mặc: ngực trái của người mặc nằm bên PHẢI file in.
# Cỡ là phần trăm khung: 26% của khung 4500 x 5100 là hộp 1170 x 1326 px, tức khoảng 10 cm,
# cỡ logo ngực quen thuộc; 55% là khoảng 21 cm, một bản in ngực giữa cỡ A4.
PRESETS: dict[str, tuple[str, float]] = {
    "full": ("center", 100.0),        # lấp đầy khung, in lưng hoặc ngực toàn khổ
    "chest-left": ("top-right", 26.0),   # logo ngực trái người mặc
    "chest-right": ("top-left", 26.0),   # logo ngực phải người mặc
    "chest": ("top", 55.0),           # ngực giữa, cỡ A4
    "back-neck": ("top", 20.0),       # nhãn nhỏ sau gáy
}


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    """`--preset` điền `--place` và `--scale` cho những cờ còn ở mặc định; cờ đặt tay vẫn thắng.

    Gọi ở đầu mỗi lần xử lý chứ không ở parse_args, để trang (dựng cờ từ dict, không qua dòng
    lệnh) đi cùng một đường với CLI."""
    if getattr(args, "preset", "none") in ("none", None):
        return args
    place, scale = PRESETS[args.preset]
    if args.place == "center":
        args.place = place
    if args.scale == 100.0:
        args.scale = scale
    return args


def art_box(box: tuple[int, int], scale: float) -> tuple[int, int]:
    """The box the design itself is fitted into: `scale` per cent of the print canvas, both ways.

    Fitting into a scaled copy of the canvas rather than to a width keeps the design's own
    proportions and makes the number mean the same thing whatever shape the design is: a chest
    print at 26 % of a 4500 x 5100 canvas lands in 1170 x 1326, and a wide design uses the width
    of that while a tall one uses the height."""
    w, h = box
    return max(1, round(w * scale / 100.0)), max(1, round(h * scale / 100.0))


def place_on_canvas(img: Image.Image, box: tuple[int, int], place: str = "center",
                    margin: float = 2.0) -> Image.Image:
    """Put the design on a transparent canvas of exactly box size, anchored where asked.

    `margin` is per cent of the short side and is the gap from the edges the design is pushed
    against, so a corner print does not sit flush against the trim. A centered design is centered
    and never sees it -- which is what a full-size print wants, and what this did before."""
    canvas = Image.new("RGBA", box, (0, 0, 0, 0))
    m = round(min(box) * margin / 100.0)
    where = place.split("-") if place != "center" else []
    if "left" in where:
        x = m
    elif "right" in where:
        x = box[0] - m - img.width
    else:
        x = (box[0] - img.width) // 2
    if "top" in where:
        y = m
    elif "bottom" in where:
        y = box[1] - m - img.height
    else:
        y = (box[1] - img.height) // 2
    canvas.paste(img, (max(0, x), max(0, y)))
    return canvas


def fit_box(w: int, h: int, box: tuple[int, int]) -> tuple[int, int]:
    """Largest (w, h) with the same aspect ratio that fits inside box."""
    scale = min(box[0] / w, box[1] / h)
    return max(1, int(w * scale)), max(1, int(h * scale))  # floor: never exceed the box


def crop_to_content(img: Image.Image, margin: float = 0.02, min_alpha: int = 1) -> Image.Image:
    """Crop to the bounding box of alpha >= min_alpha, padded by margin per side."""
    alpha = np.asarray(img.convert("RGBA"))[:, :, 3]
    ys, xs = np.nonzero(alpha >= min_alpha)
    if len(xs) == 0:
        raise EmptyResult("tách nền ra rỗng, không còn pixel nào")
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad_x, pad_y = round((x1 - x0) * margin), round((y1 - y0) * margin)
    out = Image.new("RGBA", (x1 - x0 + 2 * pad_x, y1 - y0 + 2 * pad_y), (0, 0, 0, 0))
    out.paste(img.convert("RGBA").crop((x0, y0, x1, y1)), (pad_x, pad_y))
    return out


# ---------------------------------------------------------------- color
def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0-255, Nx3) -> CIELAB (Nx3). Good enough for palette merging."""
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], axis=1)


def _merge_palette(palette: np.ndarray, counts: np.ndarray, colors: int, merge_delta_e: float = MERGE_DELTA_E) -> np.ndarray:
    """Agglomeratively merge the closest pair (in Lab) until <= colors remain and no pair is
    closer than MERGE_DELTA_E. Returns, per input entry, the index of its final group.

    Merging by distance rather than population keeps small distinct regions
    (an eye highlight, a mouth) from being absorbed by large noisy ones.
    """
    lab = _rgb_to_lab(palette)
    n = len(palette)
    centers, weights = lab.copy(), counts.astype(np.float64)
    parent = np.arange(n)
    active = np.ones(n, dtype=bool)
    D = np.linalg.norm(centers[:, None] - centers[None], axis=2)
    np.fill_diagonal(D, np.inf)
    n_active = n
    while n_active > 1:
        a, b = np.unravel_index(np.argmin(D), D.shape)
        if n_active <= colors and D[a, b] >= merge_delta_e:
            break
        centers[a] = (centers[a] * weights[a] + centers[b] * weights[b]) / (weights[a] + weights[b])
        weights[a] += weights[b]
        active[b] = False
        parent[parent == b] = a
        D[b, :] = np.inf
        D[:, b] = np.inf
        d = np.linalg.norm(centers - centers[a], axis=1)
        d[~active] = np.inf
        d[a] = np.inf
        D[a, :] = d
        D[:, a] = d
        n_active -= 1
    _, mapping = np.unique(parent, return_inverse=True)
    return mapping


def _edge_mask(rgb: np.ndarray, threshold: int = 30) -> np.ndarray:
    """True near color boundaries (anti-aliased ramps), at any resolution.

    Neighbours are sampled `step` px away (≈1/500 of the short side) so a soft ramp in a
    4500 px image is detected as firmly as a 1 px edge in a 600 px one; the mask is then
    grown by 2*step px.
    """
    h, w = rgb.shape[:2]
    step = max(1, round(min(w, h) / 500))
    a = rgb.astype(np.int16)
    p = np.pad(a, ((step, step), (step, step), (0, 0)), mode="edge")
    diff = np.zeros((h, w), dtype=np.int16)
    for dy, dx in ((0, step), (0, -step), (step, 0), (-step, 0)):
        diff = np.maximum(diff, np.abs(a - p[step + dy:h + step + dy, step + dx:w + step + dx]).sum(axis=2))
    edge = diff > threshold
    grow = 2 * step
    p2 = np.pad(edge, grow, mode="edge")
    out = edge.copy()
    for s in range(1, grow + 1):
        out |= p2[grow - s:grow - s + h, grow:grow + w] | p2[grow + s:grow + s + h, grow:grow + w]
        out |= p2[grow:grow + h, grow - s:grow - s + w] | p2[grow:grow + h, grow + s:grow + s + w]
    return out


def _erode(mask: np.ndarray, r: int) -> np.ndarray:
    """Binary erosion by a (2r+1) square, separable, no scipy."""
    out = mask.copy()
    for axis in (0, 1):
        acc = out.copy()
        for s in range(1, r + 1):
            acc &= np.roll(out, s, axis=axis) & np.roll(out, -s, axis=axis)
        out = acc
    return out


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Binary dilation by a (2r+1) square. Padded so nothing wraps around the image edge."""
    padded = np.pad(mask, r, constant_values=False)
    return ~_erode(~padded, r)[r:-r, r:-r]


def _absorb_scattered(labels: np.ndarray, final: np.ndarray, opaque: np.ndarray, max_dist: float) -> np.ndarray:
    """Relabel colors that are both spatially scattered (mostly thin specks) and chromatically
    close (< max_dist in Lab) to another color: that combination is AI grain, not a design
    color. Compact small regions and thin-but-distinct outlines are left alone.
    """
    h, w = labels.shape
    r = max(1, round(min(w, h) / 300))
    lab = _rgb_to_lab(np.rint(final))
    k = len(final)
    changed = True
    while changed:
        changed = False
        counts = np.bincount(labels[opaque], minlength=k)
        alive = counts > 0
        for i in np.argsort(counts):  # smallest first
            if not alive[i]:
                continue
            m = (labels == i) & opaque
            frac = _erode(m, r).sum() / counts[i]
            if frac >= 0.35:
                continue
            d = np.linalg.norm(lab - lab[i], axis=1)
            d[~alive] = np.inf
            d[i] = np.inf
            j = int(np.argmin(d))
            if d[j] < max_dist:
                labels[m] = j
                alive[i] = False
                changed = True
                break
    return labels


def _decontaminate_fringe(labels: np.ndarray, alpha: np.ndarray, max_iter: int = 64) -> np.ndarray:
    """Give every semi-transparent pixel the color label of its nearest fully opaque neighbour.

    Background removal leaves fringe pixels blended with the old background (a light halo on
    a dark shirt). Growing the solid colors outward into the fringe fixes that in place.
    """
    labels = labels.copy()
    fixed = alpha == 255
    todo = (alpha > 0) & ~fixed
    for _ in range(max_iter):
        if not todo.any():
            break
        newly = np.zeros_like(fixed)
        for axis, s in ((0, 1), (0, -1), (1, 1), (1, -1)):
            src_fixed = np.roll(fixed, s, axis=axis)
            take = todo & src_fixed & ~newly
            labels[take] = np.roll(labels, s, axis=axis)[take]
            newly |= take
        fixed |= newly
        todo &= ~newly
    return labels


def quantize(img: Image.Image, colors: int, binary_alpha: bool, merge_delta_e: float = MERGE_DELTA_E) -> Image.Image:
    """Reduce RGB to at most `colors` flat colors; keep alpha separately.

    1. Median-filter to kill AI grain.
    2. Bin colors at 16 levels per channel; every non-trivial bin among *interior* pixels
       (anti-aliased edges excluded) becomes a palette candidate, so a small white
       highlight gets its own entry no matter how few pixels it has.
    3. Merge candidates by perceptual distance down to `colors` (and always merge
       near-identical noise shades).
    4. Snap every pixel (edges included) to the nearest final color.
    """
    rgba = np.asarray(img.convert("RGBA")).copy()
    alpha = rgba[:, :, 3]
    opaque = alpha > 0
    rgb = rgba[:, :, :3]
    if opaque.any():
        # fill transparent pixels with the mean opaque color so they do not create their own bins
        rgb[~opaque] = rgb[opaque].mean(axis=0).astype(np.uint8)
    arr = np.asarray(Image.fromarray(rgb, "RGB").filter(ImageFilter.MedianFilter(5)))
    bins = arr >> 4
    bin_id = (bins[:, :, 0].astype(np.int32) << 8) | (bins[:, :, 1].astype(np.int32) << 4) | bins[:, :, 2]

    # Interior pixels weigh 1, anti-aliased edge pixels 0.1: blends between two colors
    # cannot pull a small highlight toward them, but a thin outline (all "edge") still
    # accumulates enough weight to earn its own palette entry.
    edge = _edge_mask(arr)
    weight = np.where(edge, 0.1, 1.0) * (alpha == 255)  # semi-transparent fringe never votes
    if not opaque.any():
        weight = np.ones_like(weight)
    flat_ids = bin_id.ravel()
    flat_w = weight.ravel()
    used, inv = np.unique(flat_ids[flat_w > 0], return_inverse=True)
    wsel = flat_w[flat_w > 0]
    counts = np.bincount(inv, weights=wsel)
    sums = np.zeros((len(used), 3))
    np.add.at(sums, inv, arr.reshape(-1, 3)[flat_w > 0].astype(np.float64) * wsel[:, None])
    means = sums / counts[:, None]
    keep = counts >= max(8.0, 0.0001 * counts.sum())  # drop stray noise bins
    if keep.sum() >= 1:
        means, counts = means[keep], counts[keep]
    if len(means) > 600:  # gradients: keep the most populated candidates
        top = np.argsort(counts)[-600:]
        means, counts = means[top], counts[top]

    mapping = _merge_palette(means, counts, colors, merge_delta_e)
    final = np.array([np.average(means[mapping == gi], axis=0, weights=counts[mapping == gi])
                      for gi in range(mapping.max() + 1)])

    # snap all 4096 possible bins (by bin centre) to the nearest final color in Lab
    grid = np.indices((16, 16, 16)).reshape(3, -1).T * 16 + 8
    d = np.linalg.norm(_rgb_to_lab(grid)[:, None] - _rgb_to_lab(np.rint(final))[None], axis=2)
    labels = np.argmin(d, axis=1)[bin_id]
    labels = _absorb_scattered(labels, final, opaque, max_dist=2 * merge_delta_e)
    labels = _decontaminate_fringe(labels, alpha)
    flat = np.rint(final).astype(np.uint8)[labels]
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

    with Image.open(png_path) as im:
        w, h = im.size
    # vtracer squares this value into an area: drop patches smaller than ~1/15000 of the
    # image (≈7x7 px at 1024). Such specks are AI noise and invisible in print anyway.
    speckle = max(2, round((w * h / 15000) ** 0.5))
    vtracer.convert_image_to_svg_py(str(png_path), str(svg_path), filter_speckle=speckle, **VTRACER_OPTS)


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
UPSCALE_MODEL = {"flat": "realesrgan-x4plus-anime", "detail": "realesrgan-x4plus", "grain": "lanczos"}
# Model ảnh chụp cho riêng thân người (thân hình đặc bật, kiểu detail). x4plus làm da mịn như sáp;
# LSDIRplusC giữ vân da tự nhiên nhưng biến vân nứt của chữ đồ họa thành lốm đốm xám (trên ảnh cầu thủ,
# mực đặc 52,7% -> 48,7%), nên chỉ dùng trong thân người, phần còn lại vẫn x4plus.
PHOTO_MODEL = "4xLSDIRplusC"
PHOTO_MODEL_URL = "https://raw.githubusercontent.com/upscayl/custom-models/main/models/"


def photo_model_ready() -> bool:
    return REALESRGAN_BIN.exists() and (BIN_DIR / "models" / f"{PHOTO_MODEL}.param").exists()


def figure_blend(base: Image.Image, photo: Image.Image, figure: Image.Image, feather: int) -> Image.Image:
    """Màu trong thân người lấy từ `photo`, ngoài thân giữ `base`; alpha luôn của `base`.

    `figure` là mặt nạ thân người của model cắt hình, ở bất kỳ cỡ nào. Nó được co vào `feather` px
    rồi làm mềm cũng chừng ấy, nên mép người và chữ sát người vẫn là bản sắc nét, chuyển dần vào
    trong thân. Hình dạng và độ phủ không đổi, chỉ màu bên trong người đổi."""
    from scipy import ndimage  # noqa: PLC0415 - heavy import kept local

    m = np.asarray(figure.convert("L").resize(base.size, Image.Resampling.BILINEAR)) >= 128
    m = ndimage.binary_erosion(m, iterations=feather)
    w = ndimage.gaussian_filter(m.astype(np.float32), feather / 2)[:, :, None]
    b = np.asarray(base).astype(np.float32)
    ph = np.asarray(photo.convert(base.mode).resize(base.size, Image.Resampling.LANCZOS)).astype(np.float32)
    out = b.copy()
    out[:, :, :3] = b[:, :, :3] * (1 - w) + ph[:, :, :3] * w
    return Image.fromarray(out.clip(0, 255).round().astype(np.uint8), base.mode)


def bleed_edges(img: Image.Image) -> Image.Image:
    """Tô phần trong suốt hẳn bằng màu của pixel đặc gần nhất; alpha giữ nguyên.

    Ảnh cắt còn mang màu nền gốc (đen) dưới chỗ trong suốt. Real-ESRGAN x4plus xử lý màu và alpha
    riêng, nên màu đen đó loang vào viền: trên ảnh cầu thủ cắt ra in áo kem, dải 1-3 px sát mép tối
    hẳn (độ sáng 125-161 so với 188 bên trong) và in thành một đường viền đen quanh người. Tô trước
    bằng màu viền thì model chỉ trộn những màu giống viền. Không đổi gì ở chỗ có mực."""
    from scipy import ndimage  # noqa: PLC0415 - heavy import kept local

    rgba = np.asarray(img.convert("RGBA")).copy()
    a = rgba[:, :, 3]
    solid = a >= 128
    clear = a == 0
    if not solid.any() or not clear.any():
        return img
    iy, ix = ndimage.distance_transform_edt(~solid, return_distances=False, return_indices=True)
    rgba[clear, :3] = rgba[iy[clear], ix[clear], :3]
    return Image.fromarray(rgba, "RGBA")


def upscale(img: Image.Image, scale: int = 4, model: str = "realesrgan-x4plus-anime") -> Image.Image:
    """Real-ESRGAN if the binary exists (anime model for flat art, x4plus for painterly), else Lanczos.

    `model="lanczos"` asks for plain resampling on purpose: halftone dots and grain come out of
    either ESRGAN model as painted fur or cracked blobs, while Lanczos keeps the dots as dots."""
    img = bleed_edges(img.convert("RGBA")) if img.mode == "RGBA" else img.convert("RGB")
    if model == "lanczos":
        return img.resize((img.width * scale, img.height * scale), Image.Resampling.LANCZOS)
    if REALESRGAN_BIN.exists():
        with tempfile.TemporaryDirectory() as td:
            src, dst = Path(td) / "in.png", Path(td) / "out.png"
            img.save(src)
            subprocess.run([str(REALESRGAN_BIN), "-i", str(src), "-o", str(dst),
                            "-n", model, "-s", str(scale),
                            "-m", str(BIN_DIR / "models")],
                           check=True, capture_output=True, text=True)
            return Image.open(dst).convert(img.mode).copy()
    print("  CẢNH BÁO: không có Real-ESRGAN trong bin/, dùng Lanczos thay thế")
    return img.resize((img.width * scale, img.height * scale), Image.Resampling.LANCZOS)


def upscale_plan(src: tuple[int, int], content: tuple[int, int], inner: tuple[int, int]) -> float | None:
    """How much to shrink `src` before the 4x model, or None to skip the model.

    A source whose 4x output fits in MODEL_MAX_PX always goes through the model unshrunk, even
    for a small placement: the model firms up edges and solid ink on the way, and skipping it
    on a 1254 px image at chest-left dropped solid ink from 72% to 50%. Only a source too big
    for that (a 4K image would come out 16K, which no step after it can hold) takes another road.
    `content` is the design's box inside `src`, which is what has to fill `inner`. Needing at
    most MODEL_MIN_GROW, the source already has the detail and the final Lanczos resize is
    enough. Otherwise it is shrunk just enough to fit the budget, unless that would leave the
    design too small to fill `inner` from 4x: then the real pixels are worth more than the
    model's sharpening, and the model is skipped."""
    shrink = (MODEL_MAX_PX / (src[0] * src[1] * UPSCALE ** 2)) ** 0.5
    if shrink >= 1.0:
        return 1.0
    need = min(inner[0] / content[0], inner[1] / content[1])
    if need <= MODEL_MIN_GROW or need / shrink > UPSCALE:
        return None
    return shrink


def enlarge(img: Image.Image, shrink: float | None, model: str) -> Image.Image:
    """Carry out upscale_plan: the image as is, or shrunk by `shrink` and then through the model."""
    if shrink is None:
        return img
    if shrink < 1.0:
        img = img.resize((round(img.width * shrink), round(img.height * shrink)), Image.Resampling.LANCZOS)
    return upscale(img, model=model)


def tighten_alpha(img: Image.Image, lo: int = 96, hi: int = 160) -> Image.Image:
    """Steepen the alpha ramp so edges stay crisp for DTF/DTG (soft alpha prints as a halo)."""
    rgba = np.asarray(img.convert("RGBA")).copy()
    a = rgba[:, :, 3].astype(np.float32)
    rgba[:, :, 3] = np.clip((a - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def harden_dots(img: Image.Image, grow: float) -> Image.Image:
    """Give halftone dots and splatter back the hard edge that Lanczos smeared, for DTF.

    Enlarging `grow` times turns every dot's edge into a ramp about `grow` px long, which prints
    as ink under 40% coverage: 28% of the ink on a halftone poster, against 10% when Real-ESRGAN
    painted the dots over. The dot's true edge is where the ramp crosses half of the dot's own
    peak, so each pixel is compared with the densest alpha within 1.4 source pixels, taken on an
    alpha blurred by half a source pixel so that Lanczos's own overshoot beside the edge does not
    count as the peak. Below 40% of it the skirt is cut, above 60% it is ink, with a smooth step
    between for anti-aliasing.

    What the edge is raised to is the typical alpha of the dot's inside around it (the pixels
    within 95% of the peak, averaged with a Gaussian, capped at the peak), never the brightest
    pixel: that is the overshoot or a speck, and raising the edge to it paints a light ring round
    every shape (a 215 disk got a 252 rim that way). Averaged rather than copied from the nearest
    inside pixel, which turns sparse spray into flat cells. Every pixel keeps its own color, alpha is never lowered inside a dot,
    dim dots stay dim, and a glow whose ramp is far longer than the window barely changes.
    Measured on real files: low coverage 28% to 7% on a halftone poster, 9% to 2% on a player
    photo with splatter, overall brightness within 1.5%."""
    from scipy import ndimage  # noqa: PLC0415 - heavy import kept local

    rgba = np.asarray(img.convert("RGBA")).copy()
    a = rgba[:, :, 3].astype(np.float32)
    r = max(2, round(1.4 * grow))
    # Đỉnh lấy trên alpha đã làm mịn nửa pixel gốc: Lanczos vọt lên sát mép (ruột 215, dải vọt
    # 240), và lấy dải vọt làm đỉnh thì cả mép bị nâng lên trên ruột thành một đường viền sáng.
    peak = ndimage.maximum_filter(ndimage.gaussian_filter(a, grow / 2), size=2 * r + 1)
    ratio = a / np.maximum(peak, 1.0)
    inside = ((ratio >= 0.95) & (a > 0)).astype(np.float32)
    weight = ndimage.gaussian_filter(inside, r / 2)
    level = np.where(weight > 1e-3, ndimage.gaussian_filter(a * inside, r / 2) / np.maximum(weight, 1e-3), peak)
    level = np.minimum(level, peak)
    t = _smoothstep(ratio, 0.4, 0.6)
    edge = (inside == 0) & (t > 0)
    alpha = np.where(inside > 0, a, 0.0)
    alpha[edge] = t[edge] * np.maximum(a[edge], level[edge])
    rgba[:, :, 3] = alpha.clip(0, 255).round().astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


HALFTONE_LPI = 30  # 10 px mỗi ô ở 300 DPI, khoảng 0,85 mm: thô vừa đủ để chấm bám keo, mịn đủ để nhìn từ xa thành dốc


def halftone_fade(img: Image.Image, lpi: float = HALFTONE_LPI, below: int = DTF_COVERAGE, min_cover: float = 0.15,
                  edge_px: int = 3, angle: float = 22.5, dpi: int = DPI, smooth_sigma: float = 4.0,
                  smooth_tol: float = 0.25) -> Image.Image:
    """--halftone-fade: glow, bóng đổ, airbrush phủ dưới 40% thành chấm halftone đặc, cho in DTF.

    Mực phủ mỏng nhận ít bột keo nên bong sau vài lần giặt; trên ảnh cầu thủ, bóng đổ lên chữ là
    phần lớn mực phủ thấp. Xưởng in đổi những vùng đó thành chấm: mỗi chấm là mực đặc, bám chắc, và
    nhìn từ xa mật độ chấm cho lại đúng độ đậm. Lưới chấm tròn xoay 22,5 độ (góc ít tạo vân moiré
    với sợi vải), `lpi` dòng mỗi inch; chấm ở ô có độ phủ c chiếm đúng c diện tích ô.

    Chấm dưới `min_cover` nhỏ tới mức không giữ được keo, nên bỏ hẳn: mép ngoài cùng của glow mất
    đi, bù lại không có bụi mực rơi khỏi bàn ép. Vùng mờ không có chỗ nào đậm tới `min_cover` thì
    không phải glow mà là vệt mờ do phóng ảnh, để nguyên.

    Chỉ chỗ mờ *mượt* mới thành chấm: nơi alpha quanh pixel lệch khỏi bản làm mịn của chính nó
    (Gauss `smooth_sigma` px, lấy trung bình bình phương trên cùng cửa sổ) quá `smooth_tol` lần độ
    phủ ở đó, tối thiểu 6 mức, là vân hạt, giữ nguyên. Xét từng pixel lẻ thì không đủ: trên vân nhiễu
    thuần, ba phần mười pixel tình cờ nằm sát trung bình và vẫn bị chấm hóa. Không
    có điều kiện này, vân xám mịn của ảnh đen trắng trên nền đen thành chấm trắng rải như tuyết trên
    mảng tối, rõ hơn mọi thứ nó định sửa. Đổi lại phần phủ thấp nằm trong vân hạt vẫn còn: trên poster
    ảnh đen trắng 26,9% chỉ xuống 23,1%, trên bóng đổ cầu thủ 2,0% xuống 1,3%. Mép khử răng cưa của mảng đặc (trong `edge_px`
    quanh mực phủ từ 40% trở lên) giữ nguyên, không thì mọi đường viền thành răng cưa chấm. Màu
    không đổi, chỉ alpha: màu đã được giải theo nền nên chấm đặc in ra đúng màu ấy."""
    from scipy import ndimage  # noqa: PLC0415 - heavy import kept local

    rgba = np.asarray(img.convert("RGBA")).copy()
    a = rgba[:, :, 3].astype(np.float32)
    fade = (a > 0) & (a < below)
    if not fade.any():
        return img
    fade &= ~ndimage.binary_dilation(a >= below, iterations=edge_px)
    # Chỉ vùng mờ có chỗ đủ đậm để thành chấm: vệt mờ 1-15/255 mà Lanczos để lại cách mép vài pixel
    # không phải glow, chấm hóa nó chỉ là xóa nó đi trên một thiết kế vốn đặc.
    labels, n = ndimage.label(fade)
    if n:
        peak = ndimage.maximum(a, labels, index=np.arange(1, n + 1))
        fade &= np.concatenate([[False], peak >= min_cover * 255])[labels]
    cell = dpi / lpi
    yy, xx = np.mgrid[: a.shape[0], : a.shape[1]].astype(np.float32)
    t = np.deg2rad(angle)
    u = (xx * np.cos(t) - yy * np.sin(t)) / cell
    v = (xx * np.sin(t) + yy * np.cos(t)) / cell
    d2 = (u - np.floor(u) - 0.5) ** 2 + (v - np.floor(v) - 0.5) ** 2   # bình phương khoảng cách tới tâm ô
    cover = a / 255.0
    dot = (np.pi * d2 < cover) & (cover >= min_cover)              # đĩa diện tích pi*r^2 = độ phủ
    soft = ndimage.gaussian_filter(a, smooth_sigma)
    rough = np.sqrt(ndimage.gaussian_filter((a - soft) ** 2, smooth_sigma))   # gồ ghề của cả vùng quanh pixel
    fade &= rough <= np.maximum(6.0, smooth_tol * soft)            # vân hạt không phải vùng mờ mượt
    rgba[:, :, 3] = np.where(fade, np.where(dot, 255, 0), rgba[:, :, 3]).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def flatten_raster(img: Image.Image, colors: int, merge_delta_e: float = MERGE_DELTA_E) -> Image.Image:
    """Tighten alpha, then quantize (includes median denoise) keeping the thin soft edge."""
    return quantize(tighten_alpha(img), colors, binary_alpha=False, merge_delta_e=merge_delta_e)


# ---------------------------------------------------------------- background keying
BG_KINDS = ("none", "black", "white", "color")
SHIRT_LABEL = {"same": "cùng màu nền", "other": "khác màu nền"}


def detect_bg(img: Image.Image) -> str:
    """What the 2 % border ring looks like: 'none' (already transparent), 'black', 'white',
    or 'color' (any other solid color). resolve_bg turns this into a processing mode."""
    ring = _border_ring(np.asarray(img.convert("RGBA")))
    if np.median(ring[:, 3]) == 0:
        return "none"
    rgb = ring[:, :3].astype(int)
    mx, mn = np.median(rgb.max(axis=1)), np.median(rgb.min(axis=1))
    if mx < 60:
        return "black"
    if mn > 200 and mx - mn < 24:
        return "white"
    return "color"


def is_flat_art(img: Image.Image, threshold: float = 0.9) -> bool:
    """True for flat graphic art -- letters, logos, badges built from areas of one solid color --
    and False for anything with gradients, texture or photographic shading.

    Measured on the picture itself, on pixels far enough from the background to be ink. Reading
    it off the keyed result instead would beg the question: a brightness key makes dark ink
    faint, so on a photograph it leaves mostly the bright smooth areas standing and those look
    flat. detect_style answers a different question (which upscaling model suits the picture)
    and is deliberately easier to satisfy; this one gates how the background is keyed, so it
    only says yes when nearly every body pixel sits in uniform color."""
    im = img.convert("RGB")
    im.thumbnail((512, 512))
    a = np.asarray(im).astype(np.float32)
    dev = np.abs(a - np.asarray(im.filter(ImageFilter.BoxBlur(2))).astype(np.float32)).sum(axis=2)
    dist = np.linalg.norm(a - np.array(bg_color(im), dtype=np.float32), axis=2)
    edge = _edge_mask(np.asarray(im.filter(ImageFilter.MedianFilter(5))), threshold=40)
    body = (dist > 60) & ~edge
    if body.sum() < 100:
        return False
    return float((dev[body] < 12).mean()) > threshold


def key_style(kind: str, flat: bool) -> str:
    """Which key a design on a solid background wants: brightness or distance.

    The black and white keys set alpha from brightness, which is what artwork that fades into
    the shirt needs -- a glow, an airbrush, a photograph's shadows all thin out smoothly. Flat
    graphic art has no such fade: every color is meant as solid ink, and brightness keying
    leaves it partly see-through (a solid red at (215, 8, 22) comes out 84 % opaque, so the
    fabric shows through the letters). For that, alpha belongs to how far the color sits from
    the shirt, which is what the color key does -- and it reads black or white as just another
    background color."""
    return "color" if flat and kind in ("black", "white") else kind


def resolve_bg(kind: str, shirt: str) -> tuple[str, bool]:
    """(processing mode, refine edge?) from the background kind and the shirt the design is for.

    shirt 'same' = shirt is the background color -> key it (alpha from color, glows fade into
    the shirt). shirt 'other' -> real cutout (rembg), plus edge refinement on white and colored
    solid backgrounds, where the model leaves background slivers along the edges. 'auto': black
    art is for dark shirts (key), everything else is cut out. The CLI default is 'same': render
    the AI image on the shirt color and the exact key path handles it without a model."""
    if kind == "none":
        return "none", False
    if shirt == "auto":
        shirt = "same" if kind == "black" else "other"
    if shirt == "same":
        return kind, False  # black / white / color key
    return "ai", kind in ("white", "color")


def choose_mode(args: argparse.Namespace, kind: str) -> tuple[str, bool]:
    """--bg (hidden override) wins; otherwise --shirt decides. Refine only for cutouts on colored bg."""
    if args.bg != "auto":
        return args.bg, args.bg == "ai" and kind in ("white", "color")
    return resolve_bg(kind, args.shirt)


def _border_ring(a: np.ndarray, frac: float = 0.02) -> np.ndarray:
    """Pixels of the outer ring (2 % of the short side), flattened to (N, C)."""
    h, w = a.shape[:2]
    t = max(2, round(min(w, h) * frac))
    return np.concatenate([a[:t].reshape(-1, *a.shape[2:]), a[-t:].reshape(-1, *a.shape[2:]),
                           a[:, :t].reshape(-1, *a.shape[2:]), a[:, -t:].reshape(-1, *a.shape[2:])])


def bg_color(img: Image.Image) -> tuple[int, int, int]:
    """Median color of the border ring: the solid background color of an AI render."""
    ring = _border_ring(np.asarray(img.convert("RGB")))
    return tuple(int(v) for v in np.median(ring, axis=0).round())


def _rim_px(shape: tuple[int, ...]) -> int:
    """Width of the anti-aliasing rim between design and background, in px (0.2 % of the short side)."""
    return max(2, round(min(shape[:2]) * 0.002))


def key_color(img: Image.Image, soft: float = 60.0, floor: float = KEY_FLOOR) -> Image.Image:
    """Turn a solid colored background (e.g. light pink) into transparency, for a shirt of that
    same color. A pixel's alpha is its RGB distance from the background relative to the strongest
    design color nearby (an anti-aliased edge that is 50 % stroke + 50 % background comes out as
    the stroke color at 50 % alpha, not as an opaque pale pixel); isolated faint marks ramp over
    `soft` above the background's noise ceiling. Color is un-premultiplied against the
    background, so printing on a shirt of the background color reproduces the original exactly.
    Design areas painted in the background color become transparent too (the shirt shows).

    `floor` is a dead zone applied last: a color closer than that to the background is shirt,
    not ink, whatever the ramp says. A near-black ellipse behind a logo on a black render still
    composites back to itself at any alpha, so on screen either choice looks right -- but a
    printer lays white underbase by alpha, and a large area at a third opacity comes out as a
    grey haze on the fabric. Cutting instead of raising the ramp's start keeps every color
    above the floor exactly as it was, edges and their recovered colors included."""
    from scipy import ndimage  # noqa: PLC0415 - heavy import kept local

    rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    bg = np.array(bg_color(img), dtype=np.float32)
    dist = np.linalg.norm(rgb - bg, axis=2)
    lo = float(np.percentile(np.linalg.norm(_border_ring(rgb) - bg, axis=1), 99)) + 6.0
    alpha = np.clip((dist - lo) / soft, 0.0, 1.0)
    # Anti-aliased rim: pixels touching the background are blends of a nearby design color with
    # the background, so their alpha is relative to the strongest color around them. Only there:
    # between two design colors (hot pink beside black) the same rule would be wrong, and it is
    # rejected anyway when the recovered color falls outside the gamut.
    k = _rim_px(dist.shape)
    near_bg = ndimage.binary_dilation(dist <= lo, iterations=k)
    win = max(7, round(min(dist.shape) * 0.006) | 1)
    ref = np.maximum(ndimage.maximum_filter(dist, size=win), lo + soft)
    rim_alpha = np.clip((dist - lo) / (ref - lo), 0.0, 1.0)
    safe_rim = np.where(rim_alpha > 0, rim_alpha, 1.0)[:, :, None]
    rim_color = (rgb - (1.0 - rim_alpha[:, :, None]) * bg) / safe_rim
    in_gamut = ((rim_color > -8.0) & (rim_color < 263.0)).all(axis=2)
    alpha = np.where(near_bg & in_gamut, rim_alpha, alpha)
    alpha = np.where(dist < floor, 0.0, alpha)  # dead zone: too close to the background to be ink
    safe = np.where(alpha > 0, alpha, 1.0)[:, :, None]
    color = np.clip((rgb - (1.0 - alpha[:, :, None]) * bg) / safe, 0, 255)
    out = np.dstack([color, alpha * 255.0]).round().astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def solid_areas(img: Image.Image, ink: np.ndarray, tol: int = 14,
                share: float = SOLID_SHARE, scale: int = 768) -> np.ndarray:
    """Weight in [0, 1] per pixel: is this an area of one flat color, or part of a fade?

    Asked as "how much of the neighbourhood is the same color as me". An area of solid ink
    answers with most of its window; a glow or an airbrushed shadow answers with the thin band
    of its own step in the ramp, and photographic shading with less still. On one mixed design
    the three come out at 0.64, 0.17 and 0.31. Measured on a thumbnail, with a window sized to
    the picture rather than to pixels -- the question is about areas, and a window wider than
    the strokes it lands on would only ever see their background. Then grown a little so the
    edge of a solid area counts with its interior, and feathered so alpha never steps.

    `ink` marks the pixels the key found something at, well clear of the background: background
    is the largest flat area of all and must stay keyed away."""
    from scipy import ndimage  # noqa: PLC0415 - heavy import kept local

    im = img.convert("RGB").copy()
    im.thumbnail((scale, scale))
    a = np.asarray(im).astype(np.int16)
    h, w = a.shape[:2]
    win = max(5, round(max(h, w) * 0.015) | 1)
    r = win // 2
    pad = np.pad(a, ((r, r), (r, r), (0, 0)), mode="edge")
    count = np.zeros((h, w), np.int16)
    for dy in range(win):
        for dx in range(win):
            count += np.abs(pad[dy:dy + h, dx:dx + w] - a).max(axis=2) <= tol
    mask = np.asarray(Image.fromarray((ink * 255).astype(np.uint8), "L").resize((w, h), Image.Resampling.BILINEAR)) >= 128
    solid = (count >= round(share * win * win)) & mask
    # r reaches the edge of a flat area, whose own window straddled the boundary and so failed
    # the test; keep stops at the ink's own anti-aliased rim, so no shape grows.
    keep = ndimage.binary_dilation(mask, iterations=2)
    weight = ndimage.binary_dilation(solid, iterations=r) & keep
    # No blur: scaling the thumbnail back up is the feather, and a blur here would pull the
    # weight below 1 inside anything narrower than its own radius -- exactly the thin lettering
    # this is meant to fill in.
    up = Image.fromarray((weight * 255).astype(np.uint8), "L").resize(img.size, Image.Resampling.BILINEAR)
    return np.asarray(up).astype(np.float32) / 255.0


def bg_rgb(img: Image.Image, bg: str) -> tuple[int, int, int]:
    """The color key_bg solves against, i.e. the shirt color the result is meant to be printed on."""
    return {"black": (0, 0, 0), "white": (255, 255, 255)}.get(bg) or bg_color(img)


def key_bg(img: Image.Image, bg: str, floor: float = KEY_FLOOR, solid: bool = True) -> Image.Image:
    """Turn a black (or white) background into transparency the way dark-shirt printers do.

    black: alpha = max(R,G,B) mapped from the background level to 255, color un-premultiplied
    so that printing the result on a black shirt reproduces the original exactly. `solid` then
    lifts flat areas of ink to full alpha (see solid_areas), which the fades are not.
    Glows and airbrush fade naturally into the shirt. Design pixels that are truly black
    become transparent too, which is correct: the shirt is black.
    white: the mirror image (alpha = 255 - min(R,G,B)), for white shirts.
    color: any other solid background, keyed by color distance (see key_color); `floor` is the
    dead zone around the background color, and applies to that path only -- the black and white
    keys already send anything near the background to nearly zero on their own.
    """
    if bg == "color":
        return key_color(img, floor=floor)
    rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    shirt = np.array(bg_rgb(img, bg), dtype=np.float32)
    key = rgb.max(axis=2) if bg == "black" else 255.0 - rgb.min(axis=2)
    lo = float(np.percentile(_border_ring(key), 99)) + 6.0  # just above the background's noise ceiling
    alpha = np.clip((key - lo) / (255.0 - lo), 0.0, 1.0)
    if solid:
        # A solid area of ink is not a fade, so it has no business being see-through: a red at
        # (195, 20, 25) would otherwise print at 76 % coverage and read thin on the fabric.
        # The lift is a gain, not an assignment: alpha there means coverage times the color's own
        # brightness, so dividing that brightness out sends the inside of the area to opaque and
        # carries its anti-aliased rim up by the same factor, keeping the edge an edge. Assigning
        # full alpha instead would slam a rim pixel at 9 % straight to opaque -- a hard, bloated,
        # speckled outline. Alpha stays 0 wherever it was 0, so no ink is invented, and color is
        # re-solved below at whatever alpha comes out, so nothing about the printed result moves.
        from scipy import ndimage  # noqa: PLC0415 - heavy import kept local

        win = max(3, round(min(key.shape) * 0.003) | 1)
        ref = ndimage.maximum_filter(key, size=win)  # the flat color a rim pixel is a fraction of
        # Only where that color is ink itself: next to nothing, the gain would be unbounded and
        # a single stray pixel just above the noise would come out opaque.
        gain = np.where(ref > lo + floor, (255.0 - lo) / np.maximum(ref - lo, 1.0), 1.0)
        w = solid_areas(img, key > lo + floor)
        alpha = alpha + w * (np.clip(alpha * gain, 0.0, 1.0) - alpha)
    shirt = np.broadcast_to(shirt, rgb.shape)
    if bg == "white":
        # A cream background is close enough to white to be keyed as white, but then white ink on
        # it is *lighter* than the background, and 255 - min says zero for it: the claws, teeth
        # and logo of a mascot vanish into the shirt. Anything clearly lighter than the
        # background is ink. Its color is solved against the real background, since that is what
        # its anti-aliased rim is blended with. On a pure white background nothing can be
        # lighter, so this changes nothing there.
        # An over-sharpened render rings every dark edge with a light line, which the upscaler
        # widens into wedges at the corners and inside narrow counters; taking that as ink
        # outlines the lettering in white. The ring is the background brightened, so it keeps the
        # background's tint, while white ink is neutral and close to 255 (on the Kentucky mascot:
        # ring chroma 12, darkest channel 243; claws and logo chroma 0, darkest channel 254). So a
        # light area counts as ink when most of it is neutral white; judging the whole area, not
        # pixel by pixel, keeps the claw's own shading and rim and drops the ring even where its
        # crest happens to reach pure white.
        from scipy import ndimage  # noqa: PLC0415 - heavy import kept local

        paper = np.array(bg_color(img), dtype=np.float32)
        light = rgb.min(axis=2) - paper.min()
        hi = float(np.percentile(_border_ring(light), 99)) + 6.0
        top = (hi + 255.0 - paper.min()) / 2.0
        lift = np.clip((light - hi) / max(top - hi, 1.0), 0.0, 1.0)
        neutral = (rgb.min(axis=2) >= 255.0 - (255.0 - paper.min()) / 4.0) & (np.ptp(rgb, axis=2) <= 6.0)
        areas, n = ndimage.label(lift > 0)
        if n:
            white = np.asarray(ndimage.mean(neutral, areas, np.arange(1, n + 1))) >= 0.5
            lift = np.where(np.concatenate(([False], white))[areas], lift, 0.0)
        shirt = np.where((lift > alpha)[:, :, None], paper, shirt)
        alpha = np.maximum(alpha, lift)
    a = alpha[:, :, None]
    color = np.clip((rgb - (1.0 - a) * shirt) / np.where(a > 0, a, 1.0), 0, 255)
    out = np.dstack([color, alpha * 255.0]).round().astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def redundant_ink(keyed: Image.Image, original: Image.Image, bg: tuple[int, int, int]) -> float:
    """Tỉ lệ mực đục nhưng in ra trùng màu áo, trên tổng diện tích mực.

    Đây là chỗ máy in phủ lót trắng rồi in đè lên vải cùng màu: vừa phí vừa nổi rõ thành mảng
    trên áo. Dùng để bắt trường hợp bật --fill-holes nhầm cho một tấm poster, khi model cắt hình
    coi cả tấm là một khối và lấp luôn nền đen giữa các chi tiết."""
    a = np.asarray(keyed.convert("RGBA")).astype(np.float32)
    alpha = a[:, :, 3]
    ink = alpha > 0
    if not ink.any():
        return 0.0
    w = (alpha / 255.0)[:, :, None]
    on_shirt = a[:, :, :3] * w + np.array(bg, dtype=np.float32) * (1 - w)
    wasted = (alpha > 200) & (np.linalg.norm(on_shirt - np.array(bg, dtype=np.float32), axis=2) < 30)
    return 100.0 * float(wasted.sum()) / float(ink.sum())


def see_through(keyed: Image.Image) -> float:
    """Phần trăm thân hình bị key làm xuyên thấu (alpha dưới 128), trên toàn bộ thân hình.

    Thân hình là vùng mực rõ (alpha trên 64) lấp kín lỗ: không cần model cắt hình, nên đo được
    cả khi chưa bật --fill-holes. Ảnh chụp người trên nền cùng màu áo thủng ở áo tối, bóng đổ,
    tóc; đó là chỗ thân hình đặc vá lại."""
    from scipy import ndimage  # noqa: PLC0415 - heavy import kept local

    im = keyed.convert("RGBA")
    im.thumbnail((512, 512))
    a = np.asarray(im)[:, :, 3]
    body = ndimage.binary_fill_holes(ndimage.binary_closing(a > 64, iterations=3))
    if not body.any():
        return 0.0
    return 100.0 * float((body & (a < 128)).sum()) / float(body.sum())


def solid_core(keyed: Image.Image, original: Image.Image, silhouette: Image.Image,
               bg: tuple[int, int, int], band: float = 0.001, thin: float = 0.006,
               floor: float = 0.0) -> Image.Image:
    """--fill-holes on the key path: cover the figure solidly, keep the key everywhere else.

    `floor` > 0 (the --fill-floor flag) leaves pixels closer than that to the background to
    the key: on a black-and-white photograph over black, trousers and deep shadow are the
    background color to within a few levels, and covering them prints a slab of black ink on
    black cloth. The default 0 covers everything the silhouette holds, as before.

    Keying alone punches through design areas painted in (or shaded down to) the background
    color: a dark jersey's folds on black, skin on a same-colored pink. Those are ink, not
    shirt. It also leaves whatever it does keep semi-transparent in proportion to brightness,
    so even a solid sleeve ends up a little see-through. `silhouette` is the segmentation
    model's alpha for the same picture, which is coverage rather than brightness: taken as the
    floor for alpha, it makes the figure opaque right out to its own soft edge.

    Two things inside the silhouette are not covered, both of them background the key already
    dropped completely (at or below its noise ceiling, so bare shirt rather than dark ink).
    The first is the rim: background just outside the silhouette, grown by r = band * short
    side, which keeps the anti-aliased edge on its own soft alpha and swallows any sliver the
    mask over-reaches into. The second is a block open to the frame border from inside the
    silhouette -- where a cut-off picture has already faded out and the model carried the body
    on to the edge of the frame anyway. Everything else the model calls figure is covered, an
    arm dissolved into the dark or a patch of skin on its own color included: connecting to
    the outside background does not make it background, since on a same-colored shirt a deep
    shadow and the shirt are the same pixels.

    The silhouette is also opened by thin * short side first. A segmentation model tracing a
    bright thin detail -- a signature stroke, a stray hair -- leaves a hairline of mask around
    it, and filling that in would print the background caught inside the hairline as solid dark
    ink. Only mask that survives as an area is taken as a figure to cover; a filament is left
    to the key, which renders the detail correctly anyway.

    Color is re-solved from the original at the final alpha, so every pixel still composites to
    the original exactly on a shirt of the background color -- figure, rim and outside glow
    alike. Only the split between alpha and color changes, never the printed result.
    """
    from scipy import ndimage  # noqa: PLC0415 - heavy import kept local

    a_key = np.asarray(keyed.convert("RGBA"))[:, :, 3].astype(np.float32)
    r = max(2, round(min(a_key.shape) * band))
    sil = np.asarray(silhouette.convert("L"))
    k = 2 * max(1, round(min(sil.shape) * thin)) + 1  # opening at the silhouette's own resolution
    kw = dict(size=k, mode="nearest")
    solid = ndimage.maximum_filter(ndimage.minimum_filter((sil >= 128).astype(np.uint8), **kw), **kw)
    cover = np.asarray(Image.fromarray(sil * solid, "L").resize(keyed.size, Image.Resampling.BILINEAR)).astype(np.float32)
    gone = a_key == 0  # the key found nothing here at all: shirt, not ink
    rim = _dilate(gone & (cover < 128), r)  # background outside the figure, plus its soft edge
    labels, _ = ndimage.label(gone & ~rim)
    ids = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    frame_block = np.isin(labels, ids[ids > 0])  # body the model carried on past the picture
    orig = np.asarray(original.convert("RGB").resize(keyed.size)).astype(np.float32)
    if floor > 0:
        # Khoảng cách tới màu nền ĐO ĐƯỢC (viền ảnh), không tới `bg` mà key giải màu: nền ảnh AI
        # thường là (19, 19, 19) chứ không đen tuyệt đối, và quần xám 27 mức chỉ cách nó 14.
        shirt = np.array(bg_color(original), dtype=np.float32)
        cover = np.where(np.linalg.norm(orig - shirt, axis=2) <= floor, 0.0, cover)
    alpha = np.maximum(a_key, np.where(rim | frame_block, 0.0, cover))
    a = (alpha / 255.0)[:, :, None]
    color = np.clip((orig - (1.0 - a) * np.array(bg, dtype=np.float32)) / np.where(a > 0, a, 1.0), 0, 255)
    return Image.fromarray(np.dstack([color, alpha]).round().astype(np.uint8), "RGBA")


def parse_ink(value: str) -> tuple[int, int, int] | None:
    """Màu mực cho --ink: `black`, `white`, hoặc mã hex `#rrggbb`. `none` = giữ nguyên màu."""
    v = value.strip().lower()
    if v in ("", "none"):
        return None
    named = {"black": (0, 0, 0), "white": (255, 255, 255)}
    if v in named:
        return named[v]
    hexa = v.lstrip("#")
    if len(hexa) != 6 or any(c not in "0123456789abcdef" for c in hexa):
        raise argparse.ArgumentTypeError(f"màu mực không hiểu: {value!r} (dùng black, white hoặc #rrggbb)")
    return tuple(int(hexa[i:i + 2], 16) for i in (0, 2, 4))


def one_ink(img: Image.Image, ink: tuple[int, int, int], shirt: tuple[int, int, int]) -> Image.Image:
    """Tách một màu: cả thiết kế in bằng đúng một màu mực, alpha là độ phủ.

    Đây là bản tách màu của thợ in lụa, không phải đổ bóng thành một khối. Mỗi pixel được hỏi nó
    đi bao xa trên đường từ **màu áo** tới **màu mực**: chỗ trùng màu áo thì không có mực, chỗ tới
    hẳn màu mực thì phủ kín, chỗ ở giữa ra độ phủ ở giữa. Nhờ vậy một cái sọ chụp ảnh trên nền
    trắng cho ra đúng mảng đen và khe hở trắng như khi làm tay trong Photoshop, thay vì thành một
    vệt đen đặc.

    Hỏi trên **ảnh đã ghép lên áo**, vì đó mới là thứ mắt thấy; màu lưu trong file đã được chia
    ngược cho alpha nên tự nó không nói lên độ đậm nhạt."""
    a = np.asarray(img.convert("RGBA")).astype(np.float32)
    w = a[:, :, 3:] / 255.0
    shirt_v = np.array(shirt, dtype=np.float32)
    ink_v = np.array(ink, dtype=np.float32)
    on_shirt = a[:, :, :3] * w + shirt_v * (1 - w)
    axis = ink_v - shirt_v
    denom = float(axis @ axis)
    if denom < 30 ** 2:
        raise EmptyResult(
            f"mực {tuple(int(v) for v in ink_v)} gần trùng màu áo {tuple(int(v) for v in shirt_v)}: "
            f"in ra sẽ không thấy gì. Chọn màu mực tương phản với nền ảnh gốc.")
    t = np.clip(((on_shirt - shirt_v) @ axis) / denom, 0.0, 1.0)
    out = np.dstack([np.broadcast_to(ink_v, a[:, :, :3].shape), t * 255.0])
    return Image.fromarray(out.round().astype(np.uint8), "RGBA")


def is_grainy(cut: Image.Image, per_1k: float = 2.5, speck_px: int = 40, min_alpha: int = 64) -> bool:
    """Halftone, splatter, grain: the ink is made of many tiny separate pieces.

    Counted on a thumbnail no larger than 1024 px: pieces under `speck_px` per 1000 ink pixels,
    ink being alpha above `min_alpha`. The threshold sits at 64, not 128, on purpose: a dark
    photograph keyed on black has skin and hair at alpha 120 +- 20, and cut at 128 that breaks
    into thousands of specks (a real portrait measured 3.1 per 1000, level with a halftone
    poster); at 64 the body stays whole. Measured on 29 real designs at 64: the four halftone
    posters and two splatter pieces sit at 2.8 to 11, portraits at or below 2.0, flat art at 0.
    Those six all came out of Real-ESRGAN as fur or blobs, so this decides whether the upscale
    is allowed to invent detail at all."""
    from scipy import ndimage  # noqa: PLC0415 - heavy import kept local

    im = cut.convert("RGBA")
    im.thumbnail((1024, 1024))
    ink = np.asarray(im)[:, :, 3] > min_alpha
    if not ink.any():
        return False
    labels, n = ndimage.label(ink)
    if not n:
        return False
    sizes = np.asarray(ndimage.sum(ink, labels, range(1, n + 1)))
    return float((sizes < speck_px).sum()) / float(ink.sum()) * 1000.0 >= per_1k


def detect_style(cut: Image.Image) -> str:
    """'grain' for halftone and splatter, 'flat' when most *design* pixels (alpha >= 128) sit in
    locally uniform color, else 'detail'. Picks the upscale: Lanczos, anime model, x4plus."""
    if is_grainy(cut):
        return "grain"
    im = cut.convert("RGBA")
    im.thumbnail((512, 512))
    rgba = np.asarray(im)
    a = rgba[:, :, :3].astype(np.float32)
    mean = np.asarray(Image.fromarray(rgba[:, :, :3]).filter(ImageFilter.BoxBlur(2))).astype(np.float32)
    local_dev = np.abs(a - mean).sum(axis=2)
    edge = _edge_mask(np.asarray(Image.fromarray(rgba[:, :, :3]).filter(ImageFilter.MedianFilter(5))), threshold=40)
    body = (rgba[:, :, 3] >= 128) & ~edge
    if _palette_size(rgba) <= 8:  # line art / 2-3 color logos: all edge, no body, still flat
        return "flat"
    if body.sum() < 100:
        return "detail"
    flat_fraction = float((local_dev[body] < 12).mean())
    return "flat" if flat_fraction > 0.75 else "detail"


def _palette_size(rgba: np.ndarray, cover: float = 0.95) -> int:
    """Number of coarse color bins (8 levels per channel) covering `cover` of the opaque pixels."""
    op = rgba[:, :, 3] >= 200
    if not op.any():
        return 0
    rgb = (rgba[:, :, :3][op] >> 5).astype(np.int32)
    counts = np.sort(np.bincount((rgb[:, 0] << 6) | (rgb[:, 1] << 3) | rgb[:, 2], minlength=512))[::-1]
    return int(np.searchsorted(np.cumsum(counts) / counts.sum(), cover) + 1)


# ---------------------------------------------------------------- background
_SESSION = None
_SESSION_LOCK = threading.Lock()  # trang có thể chạy nhiều ảnh cùng lúc; model chỉ nạp một lần


def remove_bg(img: Image.Image) -> Image.Image:
    global _SESSION
    import rembg  # heavy import kept local

    with _SESSION_LOCK:
        if _SESSION is None:
            print(f"  Nạp model {REMBG_MODEL} (lần đầu sẽ tải ~900 MB về ~/.rembg/models/)...")
            _SESSION = rembg.new_session(REMBG_MODEL)
    return rembg.remove(img.convert("RGBA"), session=_SESSION, alpha_matting=False).convert("RGBA")


def refine_edge(no_bg: Image.Image, original: Image.Image, band: float = 0.05) -> Image.Image:
    """Photoshop-style Refine Edge for the model cutout on a solid colored background.

    The model's mask is trusted deep inside (core: mask shrunk by r from its outer silhouette
    only, so holes the model left open are not widened) and ignored far outside (beyond the mask
    grown by r). In the band between, and inside holes, each pixel decides by color distance
    from the background (key_color), which removes the background blob the model kept and
    brings back thin details (lightning, splatter) it dropped. r = band * short side."""
    from scipy import ndimage  # noqa: PLC0415 - heavy import kept local

    alpha = np.asarray(no_bg.convert("RGBA"))[:, :, 3]
    r = max(1, round(min(alpha.shape) * band))
    mask = alpha >= 128
    silhouette = ndimage.binary_fill_holes(mask)
    core = _erode(np.pad(silhouette, r, constant_values=False), r)[r:-r, r:-r] & mask
    halo = _dilate(mask, r)
    # Feather the core inward over r/2 so trusted interior fades into the keyed band
    # instead of meeting it at a hard seam: w = 0 at and outside the core edge, 1 deeper in.
    blurred = np.asarray(Image.fromarray((core * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(r / 2)))
    w = np.where(core, np.clip(2.0 * blurred / 255.0 - 1.0, 0.0, 1.0), 0.0)  # never leaks across narrow gaps
    keyed = np.asarray(key_color(original.convert("RGB"), floor=0.0))  # band decides by distance: keep faint detail
    out = np.zeros_like(keyed)
    out[halo] = keyed[halo]
    orig = np.asarray(original.convert("RGBA"))
    core_alpha = (w * 255.0).round().astype(np.uint8)
    use_core = core_alpha >= out[:, :, 3]
    out[use_core, :3] = orig[use_core, :3]
    out[use_core, 3] = core_alpha[use_core]
    return Image.fromarray(out, "RGBA")


def decontaminate(cut: Image.Image, bg: tuple[int, int, int], rim_max: int = 200) -> Image.Image:
    """Photoshop's Decontaminate Colors: the model cutout keeps the original RGB, so every
    soft-edge pixel is still a blend with the background. Solve for the design color
    (rgb - (1 - a) * bg) / a and keep alpha as is, so the fringe stops printing background.
    Only the rim (alpha <= rim_max) is touched: the model gives interior pixels alpha ~250,
    which is uncertainty, not blending, and tighten_alpha makes them opaque anyway."""
    rgba = np.asarray(cut.convert("RGBA")).astype(np.float32)
    alpha = rgba[:, :, 3]
    a = alpha[:, :, None] / 255.0
    bgc = np.array(bg, dtype=np.float32)
    safe = np.where(a > 0, a, 1.0)
    color = np.clip((rgba[:, :, :3] - (1.0 - a) * bgc) / safe, 0, 255)
    rim = (alpha > 0) & (alpha <= rim_max)
    out = rgba.copy()
    out[rim, :3] = color[rim]
    return Image.fromarray(out.round().astype(np.uint8), "RGBA")


def recover_design(cut: Image.Image, original: Image.Image, floor: float = KEY_FLOOR,
                   ink: float = 60.0) -> tuple[Image.Image, float, float]:
    """Lấy lại phần thiết kế mà model cắt hình bỏ đi, trả về (ảnh, % model giữ, % giữ sau cùng).

    Model cắt hình tìm *vật thể chính*, không phải *mọi phần thiết kế*: trên ảnh cầu thủ đứng trước
    chữ WARNER, nó giữ người và bỏ hết chữ, 62% thiết kế, mà không ai được báo. Đường này chỉ chạy
    với ảnh gốc nền trơn, nên mọi pixel khác hẳn màu nền chắc chắn là mực. Bước này hợp phần model
    giữ với bản key theo khoảng cách màu nền (cùng phép key của đường áo cùng màu): ở mỗi pixel,
    bản nào đục hơn thì dùng bản đó. Chỉ thêm, không bao giờ bớt: chi tiết cùng màu nền nằm trong
    hình (mắt trắng trên nền trắng) model giữ thì vẫn giữ. Viền chữ lấy lại đã được giải màu theo
    nền, nên không mang theo quầng nền sang áo khác màu.

    "Thiết kế" để đếm là pixel cách màu nền hơn `ink`: rõ ràng là mực, không phải nhiễu nền."""
    c = np.asarray(cut.convert("RGBA")).astype(np.float32)
    keyed = np.asarray(key_color(original.convert("RGB"), floor=floor)).astype(np.float32)
    rgb = np.asarray(original.convert("RGB")).astype(np.float32)
    design = np.linalg.norm(rgb - np.array(bg_color(original), dtype=np.float32), axis=2) > ink
    take = keyed[:, :, 3] > c[:, :, 3]
    out = np.where(take[:, :, None], keyed, c)

    def share(alpha: np.ndarray) -> float:
        return 100.0 * float((design & (alpha >= 128)).sum()) / float(design.sum()) if design.any() else 100.0

    return (Image.fromarray(out.round().astype(np.uint8), "RGBA"), share(c[:, :, 3]), share(out[:, :, 3]))


def patch_tinted_holes(cut: Image.Image, original: Image.Image, min_dist: float = 15.0,
                       rim: int = 2) -> tuple[Image.Image, int]:
    """Vá lỗ kín trong hình mà bên trong là màu thiết kế, không phải màu nền. Trả về (ảnh, số lỗ vá).

    Trên poster nền hồng, mặt người có vệt da sáng gần hồng; model và bước tinh chỉnh viền coi đó là
    nền và khoét một lỗ 1927 px giữa trán, in lên áo xanh thì áo lộ qua mặt. Lòng chữ O thì cũng là
    lỗ kín nhưng đúng là nền, phải giữ trong suốt. Khác nhau ở màu bên trong: lòng chữ là màu nền
    tới mức nhiễu (viền ảnh dao động tối đa 4,5 quanh màu nền), vệt da cách nền trung vị 27. Lỗ kín
    nào có trung vị khoảng cách tới màu nền trên `min_dist` (hoặc ba lần mức nhiễu nền, nếu lớn
    hơn) thì là thiết kế bị thủng: lấp bằng màu gốc, đặc, kèm dải `rim` px quanh lỗ để không còn
    vòng viền mờ cho áo lộ qua. Chỉ dùng cho đường cắt hình (áo khác màu nền): áo cùng màu nền thì
    lỗ cho áo hiện ra vốn đúng."""
    from scipy import ndimage  # noqa: PLC0415 - heavy import kept local

    out = np.asarray(cut.convert("RGBA")).copy()
    rgb = np.asarray(original.convert("RGB"))
    bg = np.array(bg_color(original), dtype=np.float32)
    dist = np.linalg.norm(rgb.astype(np.float32) - bg, axis=2)
    noise = float(np.percentile(np.linalg.norm(_border_ring(rgb.astype(np.float32)) - bg, axis=1), 99))
    thr = max(min_dist, 3.0 * noise)
    labels, n = ndimage.label(out[:, :, 3] < 128)
    if not n:
        return cut, 0
    edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    med = ndimage.median(dist, labels, index=np.arange(1, n + 1))
    tinted = np.concatenate([[False], np.asarray(med) > thr])
    tinted[edge] = False
    patch = tinted[labels]
    if not patch.any():
        return cut, 0
    patch = ndimage.binary_dilation(patch, iterations=rim) & (out[:, :, 3] < 255)
    out[patch, :3] = rgb[patch]
    out[patch, 3] = 255
    return Image.fromarray(out, "RGBA"), int(tinted.sum())


def refine_rim(cut: Image.Image, original: Image.Image, band: float = 2.0, depth: float = 3.0,
               min_contrast: float = 30.0) -> Image.Image:
    """Ước lại độ phủ ở dải `band` px sát mép ảnh cắt, nơi model gọi là đặc nhưng màu còn pha nền.

    Mép khử răng cưa của ảnh gốc là pixel pha giữa hình và nền đen. Model hay coi cả dải đó là đặc
    (alpha 255), decontaminate chỉ sửa chỗ alpha <= 200, nên dải pha giữ nguyên màu tối; phóng 3,9
    lần nó thành viền đen 3-4 px quanh người khi in lên áo sáng. Ở mỗi pixel trong dải, so khoảng
    cách tới màu nền của nó với của màu thật phía trong (pixel đặc gần nhất cách mép quá `depth`
    px): tỉ số đó là độ phủ, như Refine Edge. Chỉ hạ alpha, không bao giờ nâng; màu được giải lại
    theo nền. Pixel ước dưới 30% giữ nguyên: ở đó mép pha và bóng đổ thật không phân biệt được.
    Chỗ mà màu phía trong cũng sát nền (dưới `min_contrast`) thì không đủ tương phản để
    đo, giữ nguyên; tóc tối có màu trong cũng tối, nên tỉ số gần 1 và mép tóc không bị khoét."""
    from scipy import ndimage  # noqa: PLC0415 - heavy import kept local

    out = np.asarray(cut.convert("RGBA")).astype(np.float32)
    rgb = np.asarray(original.convert("RGB")).astype(np.float32)
    a = out[:, :, 3]
    solid = a >= 128
    depth_in = ndimage.distance_transform_edt(solid)
    rim = solid & (depth_in <= band)
    interior = solid & (depth_in > depth)
    if not rim.any() or not interior.any():
        return cut
    iy, ix = ndimage.distance_transform_edt(~interior, return_distances=False, return_indices=True)
    bg = np.array(bg_color(original), dtype=np.float32)
    d_pix = np.linalg.norm(rgb - bg, axis=2)
    d_ref = np.linalg.norm(rgb[iy, ix] - bg, axis=2)
    est = np.clip(d_pix / np.maximum(d_ref, 1.0), 0.0, 1.0) * 255.0
    # Chỉ pixel pha rõ ràng (30% trở lên). Dưới đó, một pixel tối cạnh mảng sáng có thể là mép pha
    # 7% hay là bóng đổ thật của thiết kế: màu như nhau. Bóng dưới đế giày dày nhiều pixel, và khoét
    # 2 px ngoài cùng của nó làm mép bóng lởm chởm, áo lộ qua: tệ hơn để nguyên như model cắt.
    use = rim & (d_ref >= min_contrast) & (est < a) & (est >= 0.3 * 255.0)
    if not use.any():
        return cut
    new_a = np.where(use, est, a)
    w = (new_a / 255.0)[:, :, None]
    color = np.clip((rgb - (1.0 - w) * bg) / np.where(w > 0, w, 1.0), 0, 255)
    out[use, :3] = color[use]
    out[:, :, 3] = new_a
    return Image.fromarray(out.round().astype(np.uint8), "RGBA")


def fill_holes(cut: Image.Image, original: Image.Image) -> Image.Image:
    """Make enclosed transparent regions opaque again, restoring RGB from the original.

    Background removal deletes white details (eye highlights, teeth) that match the
    background. Only regions NOT connected to the image border are filled, so the
    surrounding background stays transparent. Intentional see-through holes (letter
    counters, rings) get filled too, which is why this is opt-in (--fill-holes).
    """
    alpha = np.asarray(cut.convert("RGBA"))[:, :, 3]
    mask = Image.fromarray(np.where(alpha < 128, 0, 255).astype(np.uint8), "L")
    padded = Image.new("L", (mask.width + 2, mask.height + 2), 0)
    padded.paste(mask, (1, 1))
    ImageDraw.floodfill(padded, (0, 0), 128)  # background connected to the border -> 128
    enclosed = np.asarray(padded)[1:-1, 1:-1] == 0
    out = np.asarray(cut.convert("RGBA")).copy()
    orig = np.asarray(original.convert("RGBA").resize(cut.size))
    out[enclosed, :3] = orig[enclosed, :3]
    out[enclosed, 3] = 255
    return Image.fromarray(out, "RGBA")


# ---------------------------------------------------------------- export
def clean_print(img: Image.Image, dpi: int = DPI, faint: int = 8,
                speck_mm: float = 0.5, speck_alpha: int = 40) -> Image.Image:
    """Bỏ mực gần như vô hình trước khi lưu file in.

    Hai thứ bị bỏ, đều là thứ mắt không thấy nhưng máy in vẫn xử lý. Một là pixel alpha dưới
    `faint`: ở mức đó lớp lót trắng không thành hình nhưng RIP vẫn tính, cho ra một lớp mờ bẩn.
    Hai là đốm vừa nhỏ hơn `speck_mm` vừa không chỗ nào vượt `speck_alpha`, tức bụi chứ không
    phải chi tiết. Ngưỡng alpha chính là thứ giữ lại hạt halftone và vệt sờn cố ý: chúng nhỏ
    nhưng đậm.

    Đo trên một poster in thật: bỏ 1,6 triệu pixel mờ mà chỉ mất 0,013% tổng lượng mực."""
    from scipy import ndimage  # noqa: PLC0415 - heavy import kept local

    a = np.asarray(img.convert("RGBA")).copy()
    alpha = a[:, :, 3]
    alpha[alpha <= faint] = 0
    ink = alpha > 0
    if ink.any():
        side = max(1, round(speck_mm / 25.4 * dpi))
        labels, n = ndimage.label(ink)
        if n:
            index = range(1, n + 1)
            sizes = np.array(ndimage.sum(ink, labels, index))
            peaks = np.array(ndimage.maximum(alpha, labels, index))
            dust = np.nonzero((sizes < side * side) & (peaks <= speck_alpha))[0] + 1
            if dust.size:
                alpha[np.isin(labels, dust)] = 0
    a[:, :, 3] = alpha
    return Image.fromarray(a, "RGBA")


META_KEY = "tshirt-pipeline"  # tên đoạn tEXt trong file PNG ghi lại cách file được tạo
RUN_ONLY = {"files", "keep_input", "ui", "audit", "icc"}  # cờ của một lần chạy, không nói gì về file


def flags_used(args: argparse.Namespace) -> dict:
    """Những cờ khác mặc định của dòng lệnh, tức những gì cần nhớ để chạy lại ra đúng file này."""
    defaults = vars(parse_args([]))
    return {k: v for k, v in vars(args).items() if k not in RUN_ONLY and v != defaults[k]}


def cmd_line(flags: dict) -> str:
    """Dựng lại dòng lệnh từ bộ cờ, để dán thẳng vào terminal."""
    parts = ["./run.sh"]
    for k, v in flags.items():
        flag = "--" + k.replace("_", "-")
        if v is True:
            parts.append(flag)
        elif v is not False and v is not None:
            parts += [flag, f"{v:g}" if isinstance(v, float) else str(v)]
    return " ".join(parts)


def read_meta(path: Path) -> dict | None:
    """Cách file in này được tạo, đọc từ chính file; None nếu file làm trước khi có mục này."""
    im = Image.open(path)
    im.load()
    raw = getattr(im, "text", {}).get(META_KEY)
    return json.loads(raw) if raw else None


# Hồ sơ CMYK để đoán màu in: của xưởng nếu có (--icc), không thì hồ sơ chung của macOS, mức bi quan
CMYK_CANDIDATES = (Path("/System/Library/ColorSync/Profiles/Generic CMYK Profile.icc"),)
CMYK_PROFILE: Path | None = next((c for c in CMYK_CANDIDATES if c.exists()), None)
# Một màu lệch quá ngần này ΔE sau khi qua CMYK là "ngoài gamut". 25 là mức xỉn hẳn: xanh lá neon
# (75), xanh dương thuần (98), tím (60), đỏ 255 (34) đều vượt; đỏ logo, hồng neon, xanh Bills lệch
# 19-20 với hồ sơ chung của macOS và chưa tính, vì hồ sơ đó hẹp hơn mực DTF thật. Muốn khắt khe hơn
# thì đưa hồ sơ của xưởng vào bằng --icc.
GAMUT_DE = 25.0
GAMUT_WARN = 30.0  # cảnh báo khi quá ngần này phần trăm mực nằm ngoài gamut
_PROOF_LOCK = threading.Lock()
_PROOF: dict[str, tuple] = {}


def _proof_transforms(icc: Path):
    """sRGB -> CMYK -> sRGB, dựng một lần cho mỗi hồ sơ."""
    from PIL import ImageCms  # noqa: PLC0415 - heavy import kept local

    key = str(icc)
    with _PROOF_LOCK:
        if key not in _PROOF:
            srgb = ImageCms.createProfile("sRGB")
            cmyk = ImageCms.getOpenProfile(key)
            intent = ImageCms.Intent.RELATIVE_COLORIMETRIC
            _PROOF[key] = (ImageCms.buildTransform(srgb, cmyk, "RGB", "CMYK", renderingIntent=intent),
                           ImageCms.buildTransform(cmyk, srgb, "CMYK", "RGB", renderingIntent=intent))
        return _PROOF[key]


def _through_cmyk(rgb: Image.Image, icc: Path) -> Image.Image:
    from PIL import ImageCms  # noqa: PLC0415 - heavy import kept local

    to_cmyk, back = _proof_transforms(icc)
    return ImageCms.applyTransform(ImageCms.applyTransform(rgb, to_cmyk), back)


def soft_proof(img: Image.Image, icc: Path | None = None) -> Image.Image:
    """Ảnh như máy in CMYK sẽ ra: màu đi qua hồ sơ CMYK rồi về sRGB, alpha giữ nguyên.

    Hồng neon, xanh lá chói, đỏ tươi của ảnh AI nằm ngoài gamut mực nên xỉn đi; đây là cách nhìn
    thấy điều đó trước khi in. Hồ sơ đúng nhất là của chính xưởng; không có thì hồ sơ chung của
    macOS cho mức bi quan."""
    icc = icc or CMYK_PROFILE
    rgba = img.convert("RGBA")
    if icc is None:
        return rgba
    rgb = _through_cmyk(rgba.convert("RGB"), icc)
    rgb.putalpha(rgba.getchannel("A"))
    return rgb


def gamut_clip(img: Image.Image, icc: Path | None = None, sample: int = 200_000) -> dict | None:
    """Phần trăm mực nằm ngoài gamut CMYK (lệch quá GAMUT_DE sau khi qua hồ sơ) và mức lệch lớn
    nhất. None khi không có hồ sơ nào để đoán. Đo trên tối đa `sample` pixel mực đục."""
    icc = icc or CMYK_PROFILE
    if icc is None:
        return None
    a = np.asarray(img.convert("RGBA"))
    idx = np.flatnonzero(a[:, :, 3] > 128)
    if not idx.size:
        return {"gamut": 0.0, "de_max": 0.0}
    if idx.size > sample:
        idx = np.random.default_rng(0).choice(idx, sample, replace=False)
    rgb = a[:, :, :3].reshape(-1, 3)[idx]
    back = np.asarray(_through_cmyk(Image.fromarray(rgb.reshape(-1, 1, 3), "RGB"), icc)).reshape(-1, 3)
    de = np.linalg.norm(_rgb_to_lab(rgb) - _rgb_to_lab(back), axis=1)
    return {"gamut": round(100.0 * float((de > GAMUT_DE).mean()), 1), "de_max": round(float(de.max()), 1)}


MIN_FEATURE_MM = 0.5  # nét mảnh hơn mức này bám keo DTF kém và bong sau vài lần giặt
SPECK_MM2 = 1.0  # đốm rời nhỏ hơn diện tích này (mm²) là hạt bụi trên bàn ép, dễ rơi


def _ink_pieces(alpha: np.ndarray, dpi: int = DPI) -> tuple[np.ndarray, np.ndarray, int]:
    """Mực đục (alpha > 128), phần của nó mảnh hơn MIN_FEATURE_MM, và bán kính co tính bằng px."""
    from scipy import ndimage  # noqa: PLC0415 - heavy import kept local

    ink = alpha > 128
    r = max(1, round(MIN_FEATURE_MM / 2 / 25.4 * dpi))
    core = ndimage.binary_opening(ink, structure=np.ones((2 * r + 1, 2 * r + 1), bool))
    return ink, ink & ~core, r


def fine_ink(img: Image.Image, dpi: int = DPI) -> dict:
    """Hai phần trăm mực mà DTF hay bong: `manh` nằm trong nét mảnh hơn 0,5 mm, `dom` là đốm rời
    nhỏ hơn 1 mm². Đo trên chính file in ở DPI của nó. Poster halftone đo được 8 đến 16% mảnh và
    2 đến 3% đốm; chữ, logo và ảnh chụp dưới 3% và 0,1%."""
    from scipy import ndimage  # noqa: PLC0415 - heavy import kept local

    alpha = np.asarray(img.convert("RGBA"))[:, :, 3]
    ink, thin, _ = _ink_pieces(alpha, dpi)
    total = float(ink.sum())
    if not total:
        return {"manh": 0.0, "dom": 0.0}
    labels, n = ndimage.label(ink)
    sizes = np.asarray(ndimage.sum(ink, labels, range(1, n + 1))) if n else np.zeros(0)
    px_mm = dpi / 25.4
    speck = float(sizes[sizes < SPECK_MM2 * px_mm * px_mm].sum()) if n else 0.0
    return {"manh": round(100.0 * float(thin.sum()) / total, 1), "dom": round(100.0 * speck / total, 2)}


def min_feature(img: Image.Image, mm: float = MIN_FEATURE_MM, dpi: int = DPI) -> Image.Image:
    """Nới mọi nét và đốm mảnh hơn `mm` ra đúng `mm`, giữ nguyên phần còn lại.

    Chỉ phần mảnh được nở ra, nên mảng khối và mép của nó không đổi. Pixel mới lấy màu và alpha
    của pixel mực gần nhất, tức nét dày lên bằng chính màu của nó, không thêm màu lạ. Đây là
    'nét tối thiểu' thợ in lụa vẫn làm tay: mất một chút chi tiết, đổi lấy áo không bong sau khi
    giặt."""
    from scipy import ndimage  # noqa: PLC0415 - heavy import kept local

    a = np.asarray(img.convert("RGBA")).copy()
    ink, thin, r = _ink_pieces(a[:, :, 3], dpi)
    if not thin.any():
        return Image.fromarray(a, "RGBA")
    grown = ndimage.binary_dilation(thin, structure=np.ones((2 * r + 1, 2 * r + 1), bool))
    new = grown & ~ink
    _, (iy, ix) = ndimage.distance_transform_edt(~ink, return_indices=True)
    a[new] = a[iy[new], ix[new]]
    return Image.fromarray(a, "RGBA")


def save_print_png(img: Image.Image, path: Path, clean: bool = True, meta: dict | None = None) -> None:
    """Lưu file in: dọn mực vô hình, gắn hồ sơ màu sRGB, ghi DPI, ghi cách file được tạo.

    sRGB phải có: RIP gặp file không gắn hồ sơ sẽ tự đoán không gian màu, và màu in ra lệch so
    với thứ đã duyệt trên màn hình. `meta` là bộ cờ và cách xử lý đã chọn, đi theo file để sau
    này mở ra là biết nó được chạy thế nào, không phải ghi chú tay sau mỗi lô."""
    from PIL import ImageCms, PngImagePlugin  # noqa: PLC0415 - heavy import kept local

    path.parent.mkdir(parents=True, exist_ok=True)
    out = clean_print(img) if clean else img.convert("RGBA")
    info = PngImagePlugin.PngInfo()
    if meta is not None:
        info.add_text(META_KEY, json.dumps(meta, ensure_ascii=False))
    out.save(path, "PNG", dpi=(DPI, DPI), optimize=False, pnginfo=info,
             icc_profile=ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())


def _checkerboard(size: tuple[int, int], cell: int = 32) -> Image.Image:
    w, h = size
    tile = np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.indices((h, w))
    tile[((yy // cell) + (xx // cell)) % 2 == 0] = 255
    tile[tile == 0] = 200
    return Image.fromarray(tile, "L").convert("RGBA")


def make_review(original: Image.Image, result: Image.Image, path: Path, height: int = 800, gap: int = 20,
                shirt: tuple[int, int, int] | None = None) -> None:
    """Original on the left, result on the right over a checkerboard (or a shirt color when keyed)."""
    def scaled(im: Image.Image) -> Image.Image:
        return im.convert("RGBA").resize((max(1, round(im.width * height / im.height)), height), Image.Resampling.LANCZOS)

    left, right = scaled(original), scaled(result)
    canvas = Image.new("RGBA", (left.width + gap + right.width, height), (255, 255, 255, 255))
    canvas.paste(left, (0, 0), left)
    board = Image.new("RGBA", right.size, shirt + (255,)) if shirt else _checkerboard(right.size)
    board.alpha_composite(right)
    canvas.paste(board, (left.width + gap, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, "PNG")


# ---------------------------------------------------------------- chấm chất lượng
def low_coverage(img: Image.Image) -> float:
    """Phần trăm diện tích mực nằm dưới 40% độ phủ, tính trên chính file in."""
    alpha = np.asarray(img.convert("RGBA"))[:, :, 3]
    ink = alpha > 0
    return 100.0 * float((alpha[ink] < DTF_COVERAGE).mean()) if ink.any() else 0.0


def measure_print(out: Path, src: Path) -> dict:
    """Ba con số chấm một file in, tính trên đúng màu áo mà file đó nhắm tới.

    `dac`: phần trăm pixel mực đặc hoàn toàn. Thấp nghĩa là mực mỏng, in ra vải lộ qua. Đọc theo
    loại thiết kế: poster halftone thấp là đúng, logo phẳng thấp là đáng ngờ.
    `phu_thap`: phần trăm mực nằm dưới 40% độ phủ. In DTF thì vùng đó nhận ít bột keo nên dễ bong
    sau vài lần giặt; in DTG có lót trắng thì không sao.
    `manh`, `dom`: phần trăm mực trong nét mảnh hơn 0,5 mm và trong đốm rời nhỏ hơn 1 mm², hai
    thứ DTF hay bong; xem fine_ink.
    `gamut`, `de_max`: phần trăm mực ngoài gamut CMYK và mức lệch lớn nhất; None khi máy không có
    hồ sơ CMYK nào. Xem gamut_clip.
    `thua`: phần trăm mực đục nhưng trùng màu áo, tức chỗ máy phủ lót trắng rồi in đè lên vải.
    `sai_so`: ghép file lên màu áo rồi so với ảnh gốc, theo mức trên 255.

    Hai số sau chỉ có nghĩa khi áo là màu nền ảnh gốc, tức đường key. File cắt hình (`--shirt
    other`) in lên áo khác màu mà ta không biết là màu gì, nên chúng là None chứ không phải một
    con số so với nhầm áo. File tách một màu cố ý khác ảnh gốc, nên riêng `sai_so` là None.
    Cách file được tạo đọc từ chính file; file làm trước khi có mục đó coi như đường key.
    """
    o = np.asarray(Image.open(out).convert("RGBA")).astype(np.float32)
    alpha = o[:, :, 3]
    ink = alpha > 0
    if not ink.any():
        return {"dac": 0.0, "phu_thap": 0.0, "manh": 0.0, "dom": 0.0, "gamut": None, "de_max": None,
                "thua": 0.0, "sai_so": None, "shirt": "#808080", "fill": False}
    rep = {
        "dac": round(100 * float((alpha[ink] > 250).mean()), 1),
        "phu_thap": round(100 * float((alpha[ink] < DTF_COVERAGE).mean()), 1),
        **fine_ink(Image.fromarray(o.astype(np.uint8), "RGBA")),
        **(gamut_clip(Image.fromarray(o.astype(np.uint8), "RGBA")) or {"gamut": None, "de_max": None}),
    }
    meta = read_meta(out) or {}
    rep["fill"] = bool(meta.get("flags", {}).get("fill_holes")) and "thân hình đặc" in meta.get("how", "")
    rep["ht"] = bool(meta.get("flags", {}).get("halftone_fade"))
    if meta.get("mode") in ("ai", "none"):
        return dict(rep, thua=None, sai_so=None, shirt=None, giu=meta.get("kept"))
    original = load_image(src).convert("RGB")
    bg = np.array(bg_color(original), dtype=np.float32)
    a = (alpha / 255.0)[:, :, None]
    on_shirt = o[:, :, :3] * a + bg * (1 - a)
    wasted = (alpha > 200) & (np.linalg.norm(on_shirt - bg, axis=2) < 30)
    kind = detect_bg(original)
    shirt = {"black": (20, 20, 22), "white": (245, 245, 245)}.get(kind) or tuple(int(v) for v in bg)
    one_color = parse_ink(str(meta.get("flags", {}).get("ink", "none"))) is not None
    return dict(rep,
                thua=round(100 * float(wasted.sum()) / float(ink.sum()), 2),
                sai_so=None if one_color else round(_key_fidelity(original, kind), 2),
                shirt="#%02x%02x%02x" % shirt)


def _key_fidelity(original: Image.Image, kind: str) -> float:
    """Sai số của riêng bước key, đo ở độ phân giải gốc nên không lẫn sai số căn ảnh."""
    if kind == "none":
        return 0.0
    mode = key_style(kind, is_flat_art(original))
    keyed = np.asarray(key_bg(original, mode)).astype(np.float32)
    a = keyed[:, :, 3:] / 255.0
    bg = np.array(bg_color(original), dtype=np.float32)
    src = np.asarray(original.convert("RGB")).astype(np.float32)
    return float(np.abs(keyed[:, :, :3] * a + bg * (1 - a) - src).mean())


# ---------------------------------------------------------------- pipeline
def process_one(src: Path, args: argparse.Namespace) -> Path:
    flags = flags_used(args)  # cờ như người dùng đặt, ghi vào file: --preset, không phải bản đã điền
    args = apply_preset(argparse.Namespace(**vars(args)))
    notes: list[str] = []  # mọi cảnh báo của lần chạy này, in ra terminal và ghi vào file cho trang hiện lại

    def warn(msg: str) -> None:
        print("  " + msg)
        notes.append(msg)
    box = target_box_px(args.size)
    inner = art_box(box, args.scale)
    original = load_image(src)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    # Model upscale được UPSCALE lần; xa hơn là Lanczos kéo giãn, và viền mềm đi. Đo trên cả
    # ảnh gốc là mức thấp nhất: hình bị cắt về khung bao rồi mới phóng, nên thực tế còn phóng hơn.
    grow = min(inner[0] / original.width, inner[1] / original.height)
    if not args.vector and grow > UPSCALE:
        times = f"{grow:.1f}".replace(".", ",")
        warn(f"CẢNH BÁO ảnh gốc nhỏ: {original.width}x{original.height} px phải phóng {times} lần "
             f"cho khung này, model chỉ làm nét được {UPSCALE} lần, phần dư là kéo giãn. "
             f"Nếu thấy mờ, sinh lại ảnh ở kích thước lớn hơn.")

    kind = detect_bg(original)
    bg, refine = choose_mode(args, kind)
    filled_in = False  # --fill-holes có thực sự được áp dụng không, để dòng log nói đúng
    figure = None  # mặt nạ thân người (thân hình đặc bật), để dùng model ảnh chụp riêng trong thân
    silhouette = None  # đường key: mặt nạ model cắt hình; None nếu không bật hoặc chốt chặn bỏ qua
    kept = None  # đường cắt hình: phần trăm thiết kế file in giữ được, ghi vào file để chấm
    style_src = None  # bức ảnh dùng để chọn kiểu upscale: bản key TRƯỚC khi tô đặc, xem bên dưới
    if bg == "none":
        cut = crop_to_content(original)
        shirt = None
    elif bg == "ai":
        no_bg = remove_bg(original)
        model_alpha = no_bg.getchannel("A")
        if refine:
            no_bg = refine_edge(no_bg, original)  # band pixels come out already decontaminated
        elif kind != "none":
            no_bg = decontaminate(no_bg, bg_color(original))
            no_bg = refine_rim(no_bg, original)
        if kind != "none":
            no_bg, by_model, kept = recover_design(no_bg, original)
            if kept - by_model > 2:
                warn(f"ĐÃ LẤY LẠI {kept - by_model:.0f}% thiết kế mà model cắt hình bỏ đi (chữ, đồ họa nằm "
                     f"rời vật thể chính): model chỉ giữ {by_model:.0f}%. Mở file in xem lại phần chữ.")
            no_bg, patched = patch_tinted_holes(no_bg, original)
            if patched:
                warn(f"ĐÃ VÁ {patched} lỗ thủng trong hình có màu thiết kế gần màu nền (da sáng, mảng nhạt) mà "
                     f"model coi là nền. Lỗ đúng màu nền như lòng chữ vẫn để trống. Mở file in xem lại.")
        if args.fill_holes:
            no_bg = fill_holes(no_bg, original)
            filled_in = True
        cut = crop_to_content(no_bg)
        # Mặt nạ thân người cắt đúng khung của cut: ghép vào kênh màu của một ảnh mang cùng alpha.
        figure = crop_to_content(Image.merge("RGBA", (model_alpha,) * 3 + (no_bg.getchannel("A"),))).getchannel("R") \
            if args.fill_holes else None
        shirt = None
    else:
        silhouette = remove_bg(original).getchannel("A") if args.fill_holes else None
        if args.bg == "auto":
            bg = key_style(bg, is_flat_art(original))  # flat art: distance, not brightness
        keyed = key_bg(original, bg, args.floor)
        # Kiểu upscale đo trên bản key thuần: tô đặc biến thân người thành một khối, còn sàn tô đặc
        # để lại nhiều mảnh nhỏ, và cả hai đều làm phép đếm hạt đổi kết quả. Cùng một ảnh phải ra
        # cùng kiểu dù bật cờ gì.
        style_src = crop_to_content(keyed, min_alpha=40) if silhouette else None
        if silhouette:
            filled = solid_core(keyed, original, silhouette, bg_rgb(original, bg), floor=args.fill_floor)
            # Trên một tấm poster, model cắt hình coi cả tấm là một khối và lấp luôn nền giữa các
            # chi tiết; mực đó in ra trùng màu áo. Đo đúng điều ấy và bỏ qua nếu vượt ngưỡng, để
            # bật nhầm cờ không làm hỏng file. Ảnh có người thật chỉ tăng 5-11%, poster tăng 27-60%.
            shirt_rgb = bg_color(original)  # nền đo được, cùng anchor với cột "mực trùng màu áo"
            added = redundant_ink(filled, original, shirt_rgb) - redundant_ink(keyed, original, shirt_rgb)
            if added > args.fill_limit:
                warn(f"BỎ QUA --fill-holes: tô đặc sẽ thêm {added:.0f}% mực in đè lên áo cùng màu "
                     f"(ngưỡng chặn tô đặc {args.fill_limit:g}%). Poster thì bỏ cờ này; ảnh có người thì "
                     f"đặt chặn tô đặc 100 (tắt chốt chặn) rồi chạy lại để tô đặc cả thân hình.")
                silhouette = None
            else:
                keyed = filled
                filled_in = True
        cut = crop_to_content(keyed, min_alpha=40)
        if not args.fill_holes and not args.vector and not is_flat_art(original) \
                and detect_style(cut) == "detail" and (holes := see_through(keyed)) > SEE_THROUGH_HINT:
            warn(f"GỢI Ý thân hình đặc: đây có vẻ là ảnh chụp người/nhân vật, và {holes:.0f}% thân hình bị "
                 f"key xuyên thấu (áo tối, bóng đổ, tóc trùng màu áo). Bật thân hình đặc (--fill-holes) "
                 f"rồi chạy lại để in thân người thành một khối liền. Poster, logo thì bỏ qua gợi ý này.")
        shirt = {"black": (20, 20, 22), "white": (245, 245, 245)}.get(kind) or bg_color(original)
    cut.save(WORK_DIR / f"{src.stem}-cut.png")

    style = detect_style(style_src or cut) if args.style == "auto" else args.style
    colors = args.colors if args.colors is not None else (12 if args.vector else 0)
    model = UPSCALE_MODEL[style]
    # Đường key phóng cả ảnh gốc rồi mới cắt; đường cắt hình phóng thẳng phần đã cắt.
    grow_src = original.size if bg not in ("ai", "none") else cut.size
    shrink = upscale_plan(grow_src, cut.size, inner)
    need = min(inner[0] / cut.width, inner[1] / cut.height)
    if args.vector:
        grow_txt = ""
    elif shrink is None:
        grow_txt = " | phóng: Lanczos, ảnh gốc đủ lớn"
        if need > MODEL_MIN_GROW:
            grow_txt = " | phóng: Lanczos, ảnh gốc quá lớn cho model"
            times = f"{need:.1f}".replace(".", ",")
            warn(f"CẢNH BÁO hình nhỏ trong ảnh gốc lớn: hình chỉ {cut.width}x{cut.height} px trong ảnh "
                 f"{original.width}x{original.height}, phải phóng {times} lần bằng Lanczos vì ảnh quá lớn "
                 f"để qua model upscale; viền có thể mềm. Cắt sát hình trước khi đưa vào để dùng model.")
    else:
        grow_txt = " | phóng: model x4" + (f", thu ảnh gốc còn {shrink:.0%}" if shrink < 1.0 else "")
    # Thân người dùng model ảnh chụp: chỉ khi người dùng đã bật thân hình đặc (nói ảnh có người), kiểu
    # là ảnh chụp, và thật sự qua model. Không có cờ đó, model cắt hình coi cả tấm poster là "người".
    photo = (style == "detail" and not args.vector and shrink is not None and args.fill_holes
             and (figure is not None or silhouette is not None) and photo_model_ready())
    if photo:
        grow_txt += f", người: {PHOTO_MODEL}"
    elif (style == "detail" and not args.vector and shrink is not None and args.fill_holes
          and (figure is not None or silhouette is not None) and REALESRGAN_BIN.exists()):
        warn(f"THIẾU MODEL {PHOTO_MODEL}: thân người phóng bằng x4plus, da sẽ mịn như sáp. Chạy lại "
             f"./setup.sh để tải model (máy vừa pull code mới thường gặp trường hợp này), rồi chạy lại ảnh.")

    if args.vector:
        q = quantize(cut, colors, binary_alpha=True, merge_delta_e=args.merge)
        q_path = WORK_DIR / f"{src.stem}-quant.png"
        q.save(q_path)
        svg_path = WORK_DIR / f"{src.stem}.svg"
        trace_svg(q_path, svg_path)
        result = render_svg(svg_path, inner)
    elif bg in ("ai", "none"):
        big = enlarge(cut, shrink, model)
        if photo:
            big = figure_blend(big, enlarge(cut, shrink, PHOTO_MODEL), figure, feather=max(4, round(big.width / cut.width * 2)))
        big = big.resize(fit_box(*big.size, inner), Image.Resampling.LANCZOS)
        result = flatten_raster(big, colors, args.merge) if colors else tighten_alpha(big)
    else:
        # keyed background: upscale the flat RGB first (cleaner edges, denoised background),
        # key at full resolution, then crop. A source that is already big enough is keyed as is.
        big_rgb = enlarge(original.convert("RGB"), shrink, model)
        if photo:
            big_rgb = figure_blend(big_rgb, enlarge(original.convert("RGB"), shrink, PHOTO_MODEL), silhouette,
                                   feather=max(4, round(big_rgb.width / original.width * 2)))
        keyed = key_bg(big_rgb, bg, args.floor)
        if silhouette:
            # Chốt chặn đo lại trên chính bản sẽ in. Ở ảnh gốc, nhiễu hạt đẩy vùng tối lên trên
            # ngưỡng "trùng màu áo"; model upscale làm mịn nó về sát nền, và một tấm ảnh đen
            # trắng từng qua chốt ở 3% rồi ra file in 36% mực đen trên vải đen.
            filled = solid_core(keyed, big_rgb, silhouette, bg_rgb(big_rgb, bg), floor=args.fill_floor)
            shirt_rgb = bg_color(original)  # nền đo được, cùng anchor với cột "mực trùng màu áo"
            added = redundant_ink(filled, big_rgb, shirt_rgb) - redundant_ink(keyed, big_rgb, shirt_rgb)
            if added > args.fill_limit:
                warn(f"BỎ QUA --fill-holes: trên bản in, tô đặc sẽ thêm {added:.0f}% mực in đè lên áo "
                     f"cùng màu (ngưỡng chặn tô đặc {args.fill_limit:g}%). Poster thì bỏ cờ này; ảnh có người "
                     f"thì đặt chặn tô đặc 100 (tắt chốt chặn) rồi chạy lại để tô đặc cả thân hình.")
                filled_in = False
            else:
                keyed = filled
        keyed = crop_to_content(keyed, min_alpha=40)
        result = keyed.resize(fit_box(*keyed.size, inner), Image.Resampling.LANCZOS)
        if colors:
            result = flatten_raster(result, colors, args.merge)
    # Halftone phóng bằng Lanczos thì mép chấm thành dốc mờ, in DTF dễ bong. Chỉ khi thật sự phóng
    # to: thu nhỏ (in sau gáy) không tạo dốc nào.
    enlarged = result.width / cut.width
    if style == "grain" and not args.vector and enlarged > 1:
        result = harden_dots(result, enlarged)
        grow_txt += ", chấm cứng"

    how = {"none": "đã trong suốt", "ai": "cắt hình" + (" + tinh chỉnh viền" if refine else ""),
           "black": "key nền đen", "white": "key nền trắng",
           "color": "key khoảng cách màu" if kind in ("black", "white") else "key màu nền"}[bg]
    how += (" + thân hình đặc" if bg != "ai" else " + lấp lỗ") if filled_in else ""
    shirt_txt = "" if bg == "none" else f" | áo: {SHIRT_LABEL['same' if bg != 'ai' else 'other']}"
    place_txt = "" if args.place == "center" and args.scale == 100.0 else f" | đặt: {args.place} {args.scale:g}%"
    place_txt += "" if args.ink.lower() in ("", "none") else f" | mực: {args.ink}"
    print(f"  nền: {kind}{shirt_txt} | cách: {how} | kiểu: {style}{grow_txt} | {'vector' if args.vector else 'raster'}{place_txt}"
          f"{f', gom {colors} màu' if colors else ', giữ nguyên màu'}")

    result = place_on_canvas(result, box, args.place, args.margin)
    ink = parse_ink(args.ink)
    if ink is not None:
        if kind == "none":
            raise EmptyResult("--ink cần biết màu áo, mà ảnh gốc đã trong suốt nên không đo được nền. "
                              "Dùng ảnh gốc có nền trơn.")
        # Màu nền thật, không phải màu áo dùng để xem: (20, 20, 22) chỉ để ảnh so sánh nhìn ra vải,
        # còn phép tách một màu phải hỏi đúng cái nền mà bước key đã giải ngược.
        result = one_ink(result, ink, bg_rgb(original, bg) if bg in BG_KINDS else bg_color(original))
    if args.halftone_fade:
        result = halftone_fade(result)
    if args.dtf_safe:
        result = min_feature(result)
    name = out_name(src.stem, box, args.place, args.scale, args.ink, args.dtf_safe,
                    fill=filled_in, cutout=(bg == "ai"), vector=args.vector, style=args.style,
                    colors=args.colors, halftone=args.halftone_fade)
    out_path = OUTPUT_DIR / name
    low = low_coverage(clean_print(result) if not args.no_clean else result)
    if low > args.dtf_warn:
        warn(f"CẢNH BÁO in DTF: {low:.0f}% diện tích mực nằm dưới 40% độ phủ (ngưỡng {args.dtf_warn:g}%). "
             f"Vùng đó nhận ít bột keo nên dễ bong; in thử một chiếc và giặt vài lần trước khi chạy số lượng. "
             f"In DTG có lót trắng thì không sao.")
    save_print_png(result, out_path, clean=not args.no_clean,
                   meta={"flags": flags, "bg": kind, "mode": bg, "how": how, "cmd": cmd_line(flags),
                         "notes": notes, **({"kept": round(kept, 1)} if kept is not None else {})})
    make_review(original, result, REVIEW_DIR / name, shirt=shirt)
    return out_path


def print_verdict(report: dict, dtf_warn: float = DTF_WARN) -> dict:
    """Kết luận một file in có dùng được không, kèm lý do. Một chỗ duy nhất quyết định điều này,
    để dòng lệnh, bảng chấm và trang không bao giờ nói khác nhau.

    Ba mức: `hong` là file không dùng được, `xem` là in được nhưng có chỗ đáng ngờ nên xem lại,
    `dat` là không thấy vấn đề nào. Ngưỡng đều lấy từ số đo thật trên một bộ 26 thiết kế."""
    hard, soft, info = [], [], []
    thua = report.get("thua")  # None = không đo được (áo khác màu nền), không phải 0
    if report["dac"] == 0 and not thua:
        hard.append("file rỗng, không có pixel mực nào")
    if thua is not None and thua > 20:
        if report.get("fill"):
            # Người dùng cố ý tô đặc thân hình: đó là quyết định in, không phải lỗi. Chỉ nói cho biết
            # phần mực đó nằm trên vải cùng màu, để họ cân nhắc chi phí mực và độ dày trên áo.
            info.append(f"{report['thua']:.0f}% mực là thân hình tô đặc trùng màu áo, in thành một khối "
                        f"liền, tốn mực hơn nhưng bám tốt. Đó là do bật thân hình đặc, đúng ý")
        else:
            hard.append(f"{report['thua']:.0f}% mực in đè lên áo cùng màu mà không bật thân hình đặc: "
                        f"nhận diện nền sai hoặc bật --fill-holes nhầm cho poster. Mở ảnh so sánh xem")
    giu = report.get("giu")
    if giu is not None and giu < 90:
        hard.append(f"model cắt hình chỉ giữ {giu:.0f}% thiết kế, mất {100 - giu:.0f}% (thường là chữ, đồ họa "
                    f"rời): file in sẽ thiếu. Mở ảnh so sánh xem, hoặc in áo cùng màu nền")
    if (report.get("sai_so") or 0) > 5:
        hard.append(f"sai số khi in {report['sai_so']:.1f} mức trên 255, mở ảnh so sánh xem bằng mắt")
    if report.get("phu_thap", 0) > dtf_warn:
        soft.append(f"{report['phu_thap']:.0f}% diện tích mực dưới 40% độ phủ, in DTF dễ bong. "
                    f"In thử một chiếc và giặt vài lần trước khi chạy số lượng")
    if report.get("manh", 0) > THIN_WARN:
        soft.append(f"{report['manh']:.0f}% mực nằm trong nét mảnh hơn {f'{MIN_FEATURE_MM:g}'.replace('.', ',')} mm, in DTF dễ bong. "
                    f"Chạy lại với --dtf-safe để nới nét, hoặc in thử rồi giặt")
    if report.get("dom", 0) > SPECK_WARN and report.get("ht"):
        # Đốm là chấm halftone người dùng bật cho vùng mờ: quyết định in, không phải lỗi.
        info.append(f"{report['dom']:.1f}% mực là chấm halftone nhỏ hơn {SPECK_MM2:g} mm² do bật chấm hóa vùng mờ, "
                    f"đúng ý. In thử để chắc chấm bám")
    elif report.get("dom", 0) > SPECK_WARN:
        soft.append(f"{report['dom']:.1f}% mực là đốm rời nhỏ hơn {SPECK_MM2:g} mm², dễ rơi khỏi bàn ép. "
                    f"--dtf-safe nới đốm ra, hoặc chấp nhận mất vài đốm")
    if (report.get("gamut") or 0) > GAMUT_WARN:
        soft.append(f"{report['gamut']:.0f}% mực nằm ngoài gamut mực CMYK, lệch tới {report['de_max']:.0f} ΔE: "
                    f"in ra xỉn hẳn so với màn hình. Bật 'xem như in' trên trang để thấy trước, hoặc đổi màu")
    if report["dac"] < 50:
        soft.append("mực mỏng, đúng với poster halftone nhưng đáng ngờ với đồ họa phẳng")
    muc = "hong" if hard else ("xem" if soft else "dat")
    return {"muc": muc,
            "nhan": {"hong": "Không dùng được", "xem": "In được, nên xem lại",
                     "dat": "Đủ điều kiện in"}[muc],
            "why": hard + soft, "info": info}


def audit_rows(sources: list[Path]) -> list[dict]:
    """Chấm mọi file in của các ảnh gốc đã cho, kèm lý do cần xem lại, sắp xếp mực mỏng lên trước.

    Kết luận và ngưỡng nằm ở print_verdict, dùng chung với trang."""
    rows = []
    for src in sources:
        for out in sorted(OUTPUT_DIR.glob(f"{glob.escape(src.stem)}_*.png")):
            r = dict(measure_print(out, src), name=out.name, src=src.name)
            r.update(print_verdict(r))
            rows.append(r)
    return sorted(rows, key=lambda r: r["dac"])


def audit(sources: list[Path]) -> int:
    """In bảng chấm cho mọi file in đang có. Trang UI cho từng ảnh; bảng này cho cả lô một lượt."""
    rows = audit_rows(sources)
    if not rows:
        print(f"Chưa có file in nào trong {OUTPUT_DIR}")
        return 0
    def num(v: float | None, width: int, digits: int) -> str:
        return f"{v:{width}.{digits}f}" if v is not None else f"{'—':>{width}s}"

    def mean(vals: list[float | None]) -> str:
        known = [v for v in vals if v is not None]
        return f"{sum(known) / len(known):.2f}" if known else "—"

    print(f"{'đặc%':>6s} {'phủ thấp%':>10s} {'mảnh%':>6s} {'đốm%':>5s} {'gamut%':>7s} {'thừa%':>7s} {'sai số':>7s}  file")
    for r in rows:
        mark = {"dat": "", "xem": "  <-- xem lại", "hong": "  <-- KHÔNG DÙNG ĐƯỢC"}[r["muc"]]
        print(f"{r['dac']:6.1f} {r['phu_thap']:10.1f} {r.get('manh', 0):6.1f} {r.get('dom', 0):5.1f} "
              f"{num(r.get('gamut'), 7, 1)} {num(r['thua'], 7, 2)} {num(r['sai_so'], 7, 2)}  {r['name'][:36]}{mark}")
    d = [r["dac"] for r in rows]
    print(f"\n{len(rows)} file | mực đặc tb {sum(d)/len(d):.1f}% | mực thừa tb {mean([r['thua'] for r in rows])}% | "
          f"sai số tb {mean([r['sai_so'] for r in rows])}  (— = không đo được, áo khác màu nền)")
    ok = sum(1 for r in rows if r["muc"] == "dat")
    print(f"đủ điều kiện in: {ok}/{len(rows)} file")
    for r in rows:
        if r["why"] or r.get("info"):
            print(f"  {r['nhan']}: {r['name'][:56]}")
            for w in r["why"]:
                print(f"      - {w}")
            for w in r.get("info", []):
                print(f"      · {w}")
    return 0


def _collect(args: argparse.Namespace) -> list[Path]:
    if args.files:
        return [Path(f) for f in args.files]
    return sorted(p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _move(src: Path, sub: str) -> None:
    dest = INPUT_DIR / sub
    dest.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest / src.name))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = p = argparse.ArgumentParser(
        description="Ảnh thiết kế AI -> file in PNG nền trong suốt, 300 DPI. Thả ảnh vào input/ rồi chạy.",
        epilog="Ví dụ: ./run.sh (áo cùng màu nền ảnh) | ./run.sh --shirt other (áo khác màu) | ./run.sh --size 30x40")
    daily = p.add_argument_group("hằng ngày")
    daily.add_argument("--shirt", choices=["same", "other", "auto"], default="same",
                       help="in lên áo màu gì so với nền ảnh. same (mặc định) = áo cùng màu nền: nền thành trong suốt bằng công thức, "
                            "chính xác từng pixel, không cần model. other = áo khác màu: cắt hình bằng model, nền trắng/màu được tinh chỉnh viền. "
                            "auto = nền đen coi là áo đen, còn lại coi là áo khác màu")
    daily.add_argument("--size", default=DEFAULT_SIZE,
                       help=f"kích thước file in, dạng WxH: pixel (vd 4500x5100) hoặc cm nếu số nhỏ hơn 200 (vd 30x40). Mặc định {DEFAULT_SIZE}")
    daily.add_argument("--place", choices=PLACES, default="center",
                       help="đặt thiết kế ở đâu trên khung in. Mặc định center. Dùng kèm --scale khi in hình nhỏ, "
                            "ví dụ --place top-right --scale 26 cho một hình nhỏ góc trên phải")
    daily.add_argument("--scale", type=float, default=100.0,
                       help="thiết kế chiếm bao nhiêu phần trăm khung in, giữ nguyên tỉ lệ hình. Mặc định 100 = lấp đầy khung")
    daily.add_argument("--preset", choices=["none", *PRESETS], default="none",
                       help="vị trí in hay dùng, thay cho --place và --scale: "
                            + ", ".join(f"{k} = {v[0]} {v[1]:g}%%" for k, v in PRESETS.items())
                            + ". Tên theo người mặc: chest-left là ngực trái người mặc, nằm bên phải file. "
                              "--place hoặc --scale đặt tay vẫn thắng")
    daily.add_argument("--vector", action="store_true",
                       help="minh họa phẳng: gom màu rồi trace vector (mảng màu tuyệt đối phẳng, viền cong mượt). Mặc định là raster: upscale AI, giữ nguyên màu")
    daily.add_argument("--ink", default="none",
                       help="in một màu duy nhất: black, white, hoặc #rrggbb. Alpha thành độ phủ, "
                            "đúng kiểu tách một màu để in lụa. Mặc định none = giữ nguyên màu")
    daily.add_argument("--audit", action="store_true",
                       help="in bảng chấm chất lượng cho mọi file in đang có rồi thoát, không xử lý ảnh nào")
    daily.add_argument("--ui", action="store_true",
                       help="mở trang xem tại máy: chọn cờ theo từng ảnh, xem kết quả trên đúng màu áo, kèm số chấm chất lượng")
    daily.add_argument("files", nargs="*", help="chỉ xử lý các file này thay cho cả input/")
    p = p.add_argument_group("nâng cao (thường không cần)")
    p.add_argument("--bg", choices=["auto", "black", "white", "color", "ai", "none"], default="auto", help=argparse.SUPPRESS)
    p.add_argument("--style", choices=["auto", "flat", "detail", "grain"], default="auto",
                   help="chọn cách upscale: flat = tranh phẳng (model anime), detail = tranh có gradient/texture "
                        "(model x4plus), grain = halftone, chấm bi, vệt bắn (Lanczos, không cho model bịa chi tiết). "
                        "auto = tự nhận diện")
    p.add_argument("--colors", type=int, default=None,
                   help="gom về tối đa N màu. Mặc định: 12 khi --vector, không gom khi raster. 0 = không gom")
    p.add_argument("--merge", type=float, default=MERGE_DELTA_E,
                   help=f"ngưỡng gộp màu gần nhau (CIELAB ΔE, mặc định {MERGE_DELTA_E:g}). Tăng nếu còn đốm màu lệch, giảm nếu hai màu khác nhau bị gộp")
    p.add_argument("--fill-limit", type=float, default=FILL_LIMIT,
                   help=f"ngưỡng an toàn cho --fill-holes: nếu tô đặc làm tăng quá ngần này phần trăm mực in đè lên "
                        f"áo cùng màu thì bỏ qua và cảnh báo. Mặc định {FILL_LIMIT:g}. 100 = tắt chốt chặn")
    p.add_argument("--dtf-warn", type=float, default=DTF_WARN,
                   help=f"cảnh báo khi quá ngần này phần trăm diện tích mực nằm dưới 40%% độ phủ, mức mà in DTF "
                        f"dễ bong. Mặc định {DTF_WARN:g}. 100 = tắt cảnh báo")
    p.add_argument("--icc", default=None,
                   help="hồ sơ màu CMYK (.icc) của xưởng in, dùng để đo màu ngoài gamut và xem như in. "
                        f"Mặc định: {'hồ sơ chung của macOS' if CMYK_PROFILE else 'không có, bỏ qua phép đo'}")
    p.add_argument("--dtf-safe", action="store_true",
                   help=f"nới mọi nét và đốm mảnh hơn {MIN_FEATURE_MM:g} mm ra đúng {MIN_FEATURE_MM:g} mm bằng chính màu của nó, "
                        "để in DTF không bong. Mất một chút chi tiết ở halftone và vệt bắn. Tên file thêm _dtf-safe")
    p.add_argument("--halftone-fade", action="store_true",
                   help=f"đổi glow, bóng đổ, airbrush phủ dưới 40%% thành chấm halftone đặc ({HALFTONE_LPI} LPI) để in DTF "
                        "bám keo. Nhìn gần thấy chấm; chấm dưới 15%% bị bỏ. Tên file thêm _ht")
    p.add_argument("--no-clean", action="store_true",
                   help="không dọn mực vô hình trước khi lưu (mặc định có dọn: bỏ alpha dưới 8 và các đốm "
                        "nhỏ hơn 0,5mm mà không chỗ nào đậm quá 40)")
    p.add_argument("--margin", type=float, default=2.0,
                   help="khoảng cách từ mép khung tới thiết kế khi --place không phải center, tính theo phần trăm cạnh ngắn. Mặc định 2")
    p.add_argument("--floor", type=float, default=KEY_FLOOR,
                   help=f"key màu nền: màu cách nền dưới ngưỡng này (khoảng cách RGB) coi như màu áo, cho trong suốt hẳn. "
                        f"Mặc định {KEY_FLOOR:g}, dập quầng xám mà máy in vẫn phủ lót trắng. 0 = tắt")
    p.add_argument("--fill-floor", type=float, default=0.0,
                   help="sàn cho --fill-holes: chỗ ảnh gốc cách màu nền dưới ngưỡng này (khoảng cách RGB) không tô đặc, "
                        "để áo làm màu đó. Dùng cho ảnh đen trắng trên nền đen, khi quần và bóng sâu gần như cùng màu nền: "
                        "thử 32. Mặc định 0 = tô đặc cả thân hình như trước")
    p.add_argument("--fill-holes", action="store_true",
                   help="không đục lỗ trong hình. Cắt hình: lấp vùng trong suốt bị bao kín (chấm sáng, răng bị model khoét). "
                        "Key (--shirt same): thân hình theo model cắt hình được giữ đặc (bóng áo tối, da trùng màu nền), ngoài thân hình vẫn key. "
                        "Không dùng nếu thiết kế có lỗ xuyên cố ý")
    p.add_argument("--keep-input", action="store_true", help="không chuyển ảnh gốc sang input/done/")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    global CMYK_PROFILE
    args = parse_args(argv)
    if args.icc:
        icc = Path(args.icc)
        if not icc.is_file():
            print(f"Không tìm thấy hồ sơ màu {icc}")
            return 1
        CMYK_PROFILE = icc
    if args.ui:
        import ui  # noqa: PLC0415 - chỉ nạp khi cần, để chạy dòng lệnh không phải nạp thêm gì

        ui.serve()
        return 0
    if args.audit:
        done = INPUT_DIR / "done"
        srcs = _collect(args) if args.files else sorted(
            p for folder in (INPUT_DIR, done) if folder.is_dir()
            for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        return audit(srcs)
    check_tools()
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = _collect(args)
    if not files:
        print(f"Không có ảnh nào trong {INPUT_DIR}")
        return 0
    box = target_box_px(args.size)
    print(f"{len(files)} ảnh | {'vector' if args.vector else 'raster'} | file in {box[0]}x{box[1]} px "
          f"@ {DPI} DPI = {box[0]/DPI*2.54:.1f} x {box[1]/DPI*2.54:.1f} cm")

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
            # --keep-input giữ ảnh tại chỗ cả khi lỗi: trên trang, ảnh bị chuyển sang failed/ sẽ
            # biến mất khỏi danh sách và không sửa cờ chạy lại được.
            if src.parent == INPUT_DIR and not args.keep_input:
                _move(src, "failed")

    print(f"\nXong: {len(ok)} thành công, {len(failed)} lỗi.")
    for src, reason in failed:
        print(f"  {src.name}: {reason}")
    if ok:
        print(f"Kết quả trong {OUTPUT_DIR}, ảnh so sánh trong {REVIEW_DIR}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
