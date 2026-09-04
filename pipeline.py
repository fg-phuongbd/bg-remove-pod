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
DEFAULT_SIZE = "4500x5100"  # px; Printful/Merch-style print file (38.1 x 43.2 cm at 300 DPI)
REMBG_MODEL = "birefnet-general"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
MERGE_DELTA_E = 12.0  # default for --merge: palette colors closer than this (CIELAB) are always merged
VTRACER_OPTS = dict(
    colormode="color", hierarchical="stacked", mode="spline",
    color_precision=8, corner_threshold=60, path_precision=3,
)  # filter_speckle is computed per image in trace_svg


class EmptyResult(Exception):
    """Background removal left no visible pixels."""


# ---------------------------------------------------------------- geometry
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


def place_on_canvas(img: Image.Image, box: tuple[int, int]) -> Image.Image:
    """Center img on a transparent canvas of exactly box size (img must already fit)."""
    canvas = Image.new("RGBA", box, (0, 0, 0, 0))
    canvas.paste(img, ((box[0] - img.width) // 2, (box[1] - img.height) // 2))
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
def upscale(img: Image.Image, scale: int = 4, model: str = "realesrgan-x4plus-anime") -> Image.Image:
    """Real-ESRGAN if the binary exists (anime model for flat art, x4plus for painterly), else Lanczos."""
    img = img.convert("RGBA") if img.mode == "RGBA" else img.convert("RGB")
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


def tighten_alpha(img: Image.Image, lo: int = 96, hi: int = 160) -> Image.Image:
    """Steepen the alpha ramp so edges stay crisp for DTF/DTG (soft alpha prints as a halo)."""
    rgba = np.asarray(img.convert("RGBA")).copy()
    a = rgba[:, :, 3].astype(np.float32)
    rgba[:, :, 3] = np.clip((a - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def flatten_raster(img: Image.Image, colors: int, merge_delta_e: float = MERGE_DELTA_E) -> Image.Image:
    """Tighten alpha, then quantize (includes median denoise) keeping the thin soft edge."""
    return quantize(tighten_alpha(img), colors, binary_alpha=False, merge_delta_e=merge_delta_e)


# ---------------------------------------------------------------- background keying
def detect_bg(img: Image.Image) -> str:
    """Look at the 2 % border ring: 'none' (already transparent), 'black' (dark shirt art),
    'white', or 'ai' (use rembg)."""
    rgba = np.asarray(img.convert("RGBA"))
    h, w = rgba.shape[:2]
    t = max(2, round(min(w, h) * 0.02))
    alpha_ring = np.concatenate([rgba[:t, :, 3].ravel(), rgba[-t:, :, 3].ravel(), rgba[:, :t, 3].ravel(), rgba[:, -t:, 3].ravel()])
    if np.median(alpha_ring) == 0:
        return "none"
    a = rgba[:, :, :3]
    ring = np.concatenate([a[:t].reshape(-1, 3), a[-t:].reshape(-1, 3), a[:, :t].reshape(-1, 3), a[:, -t:].reshape(-1, 3)])
    mx, mn = np.median(ring.max(axis=1)), np.median(ring.min(axis=1))
    if mx < 60:
        return "black"
    if mn > 200:
        return "white"
    return "ai"


def key_bg(img: Image.Image, bg: str) -> Image.Image:
    """Turn a black (or white) background into transparency the way dark-shirt printers do.

    black: alpha = max(R,G,B) mapped from the background level to 255, color un-premultiplied
    (c / alpha) so that printing the result on a black shirt reproduces the original exactly.
    Glows and airbrush fade naturally into the shirt. Design pixels that are truly black
    become transparent too, which is correct: the shirt is black.
    white: the mirror image (alpha = 255 - min(R,G,B)), for white shirts.
    """
    rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    h, w = rgb.shape[:2]
    if bg == "white":
        rgb = 255.0 - rgb  # solve as black, then invert the colors back
    key = rgb.max(axis=2)
    t = max(2, round(min(w, h) * 0.02))
    ring = np.concatenate([key[:t].ravel(), key[-t:].ravel(), key[:, :t].ravel(), key[:, -t:].ravel()])
    lo = float(np.percentile(ring, 99)) + 6.0  # just above the background's noise ceiling
    alpha = np.clip((key - lo) / (255.0 - lo), 0.0, 1.0)
    safe = np.where(alpha > 0, alpha, 1.0)
    color = np.clip(rgb / safe[:, :, None], 0, 255)  # un-premultiply: color * alpha == original
    if bg == "white":
        color = 255.0 - color
    out = np.dstack([color, alpha * 255.0]).round().astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def detect_style(cut: Image.Image) -> str:
    """'flat' when most *design* pixels (alpha >= 128) sit in locally uniform color, else 'detail'.
    Used to pick the upscale model (anime for flat art, x4plus for painterly)."""
    im = cut.convert("RGBA")
    im.thumbnail((512, 512))
    rgba = np.asarray(im)
    a = rgba[:, :, :3].astype(np.float32)
    mean = np.asarray(Image.fromarray(rgba[:, :, :3]).filter(ImageFilter.BoxBlur(2))).astype(np.float32)
    local_dev = np.abs(a - mean).sum(axis=2)
    edge = _edge_mask(np.asarray(Image.fromarray(rgba[:, :, :3]).filter(ImageFilter.MedianFilter(5))), threshold=40)
    body = (rgba[:, :, 3] >= 128) & ~edge
    if body.sum() < 100:
        return "detail"
    flat_fraction = float((local_dev[body] < 12).mean())
    return "flat" if flat_fraction > 0.75 else "detail"


# ---------------------------------------------------------------- background
_SESSION = None


def remove_bg(img: Image.Image) -> Image.Image:
    global _SESSION
    import rembg  # heavy import kept local

    if _SESSION is None:
        print(f"  Nạp model {REMBG_MODEL} (lần đầu sẽ tải ~900 MB về ~/.rembg/models/)...")
        _SESSION = rembg.new_session(REMBG_MODEL)
    return rembg.remove(img.convert("RGBA"), session=_SESSION, alpha_matting=False).convert("RGBA")


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


# ---------------------------------------------------------------- pipeline
def process_one(src: Path, args: argparse.Namespace) -> Path:
    box = target_box_px(args.size)
    original = Image.open(src)
    original.load()
    original = original.convert("RGBA")
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    bg = detect_bg(original) if args.bg == "auto" else args.bg
    if bg == "none":
        cut = crop_to_content(original)
        shirt = None
    elif bg == "ai":
        no_bg = remove_bg(original)
        if args.fill_holes:
            no_bg = fill_holes(no_bg, original)
        cut = crop_to_content(no_bg)
        shirt = None
    else:
        cut = crop_to_content(key_bg(original, bg), min_alpha=40)
        shirt = (20, 20, 22) if bg == "black" else (245, 245, 245)
    cut.save(WORK_DIR / f"{src.stem}-cut.png")

    style = detect_style(cut) if args.style == "auto" else args.style
    colors = args.colors if args.colors is not None else (12 if args.vector else 0)
    model = "realesrgan-x4plus-anime" if style == "flat" else "realesrgan-x4plus"
    print(f"  nền: {bg} | kiểu: {style} | {'vector' if args.vector else 'raster'}"
          f"{f', gom {colors} màu' if colors else ', giữ nguyên màu'}")

    if args.vector:
        q = quantize(cut, colors, binary_alpha=True, merge_delta_e=args.merge)
        q_path = WORK_DIR / f"{src.stem}-quant.png"
        q.save(q_path)
        svg_path = WORK_DIR / f"{src.stem}.svg"
        trace_svg(q_path, svg_path)
        result = render_svg(svg_path, box)
    elif bg in ("ai", "none"):
        big = upscale(cut, model=model)
        big = big.resize(fit_box(*big.size, box), Image.Resampling.LANCZOS)
        result = flatten_raster(big, colors, args.merge) if colors else tighten_alpha(big)
    else:
        # keyed background: upscale the flat RGB first (cleaner edges, denoised background),
        # key at full resolution, then crop
        big_rgb = upscale(original.convert("RGB"), model=model)
        keyed = crop_to_content(key_bg(big_rgb, bg), min_alpha=40)
        result = keyed.resize(fit_box(*keyed.size, box), Image.Resampling.LANCZOS)
        if colors:
            result = flatten_raster(result, colors, args.merge)

    result = place_on_canvas(result, box)
    out_path = OUTPUT_DIR / f"{src.stem}.png"
    save_print_png(result, out_path)
    make_review(original, result, REVIEW_DIR / f"{src.stem}.png", shirt=shirt)
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
    p.add_argument("--bg", choices=["auto", "black", "white", "ai", "none"], default="auto",
                   help="nền của ảnh gốc: black/white = chuyển độ sáng thành trong suốt (in áo tối/sáng), ai = tách nền bằng model, none = ảnh đã trong suốt. auto = tự nhận diện")
    p.add_argument("--vector", action="store_true",
                   help="minh họa phẳng: gom màu rồi trace vector (mảng màu tuyệt đối phẳng, viền cong mượt). Mặc định là raster: upscale AI, giữ nguyên màu")
    p.add_argument("--style", choices=["auto", "flat", "detail"], default="auto",
                   help="chọn model upscale: flat = tranh phẳng, detail = tranh có gradient/texture. auto = tự nhận diện")
    p.add_argument("--colors", type=int, default=None,
                   help="gom về tối đa N màu. Mặc định: 12 khi --vector, không gom khi raster. 0 = không gom")
    p.add_argument("--merge", type=float, default=MERGE_DELTA_E,
                   help=f"ngưỡng gộp màu gần nhau (CIELAB ΔE, mặc định {MERGE_DELTA_E:g}). Tăng nếu còn đốm màu lệch, giảm nếu hai màu khác nhau bị gộp")
    p.add_argument("--size", default=DEFAULT_SIZE,
                   help=f"kích thước file in, dạng WxH: pixel (vd 4500x5100) hoặc cm nếu số nhỏ hơn 200 (vd 30x40). Mặc định {DEFAULT_SIZE}")
    p.add_argument("--fill-holes", action="store_true",
                   help="lấp các vùng trong suốt bị bao kín (chấm sáng, răng bị khoét). Không dùng nếu thiết kế có lỗ xuyên cố ý")
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
    box = target_box_px(args.size)
    print(f"{len(files)} ảnh | {'vector' if args.vector else 'raster'} | file in {box[0]}x{box[1]} px @ {DPI} DPI")

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
