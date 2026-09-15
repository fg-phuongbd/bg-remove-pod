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
import threading
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
FILL_LIMIT = 20.0  # --fill-holes bị bỏ qua nếu nó thêm quá ngần này phần trăm mực trùng màu áo
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
def out_name(stem: str, box: tuple[int, int], place: str, scale: float) -> str:
    """Tên file in: tên ảnh, khung in, vị trí, và cỡ nếu không phải 100%.

    Cỡ chỉ xuất hiện khi khác 100 để tên mặc định vẫn gọn (`name_4500x5100_center.png`), nhưng
    hai bản in nhỏ cùng một góc mà khác cỡ thì không đè lên nhau."""
    tail = "" if scale == 100.0 else f"_{scale:g}pc"
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


def solid_core(keyed: Image.Image, original: Image.Image, silhouette: Image.Image,
               bg: tuple[int, int, int], band: float = 0.001, thin: float = 0.006) -> Image.Image:
    """--fill-holes on the key path: cover the figure solidly, keep the key everywhere else.

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
    alpha = np.maximum(a_key, np.where(rim | frame_block, 0.0, cover))
    a = (alpha / 255.0)[:, :, None]
    orig = np.asarray(original.convert("RGB").resize(keyed.size)).astype(np.float32)
    color = np.clip((orig - (1.0 - a) * np.array(bg, dtype=np.float32)) / np.where(a > 0, a, 1.0), 0, 255)
    return Image.fromarray(np.dstack([color, alpha]).round().astype(np.uint8), "RGBA")


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
    inner = art_box(box, args.scale)
    original = Image.open(src)
    original.load()
    original = original.convert("RGBA")
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    kind = detect_bg(original)
    bg, refine = choose_mode(args, kind)
    filled_in = False  # --fill-holes có thực sự được áp dụng không, để dòng log nói đúng
    if bg == "none":
        cut = crop_to_content(original)
        shirt = None
    elif bg == "ai":
        no_bg = remove_bg(original)
        if refine:
            no_bg = refine_edge(no_bg, original)  # band pixels come out already decontaminated
        elif kind != "none":
            no_bg = decontaminate(no_bg, bg_color(original))
        if args.fill_holes:
            no_bg = fill_holes(no_bg, original)
            filled_in = True
        cut = crop_to_content(no_bg)
        shirt = None
    else:
        silhouette = remove_bg(original).getchannel("A") if args.fill_holes else None
        if args.bg == "auto":
            bg = key_style(bg, is_flat_art(original))  # flat art: distance, not brightness
        keyed = key_bg(original, bg, args.floor)
        if silhouette:
            filled = solid_core(keyed, original, silhouette, bg_rgb(original, bg))
            # Trên một tấm poster, model cắt hình coi cả tấm là một khối và lấp luôn nền giữa các
            # chi tiết; mực đó in ra trùng màu áo. Đo đúng điều ấy và bỏ qua nếu vượt ngưỡng, để
            # bật nhầm cờ không làm hỏng file. Ảnh có người thật chỉ tăng 5-11%, poster tăng 27-60%.
            added = redundant_ink(filled, original, bg_rgb(original, bg)) - \
                redundant_ink(keyed, original, bg_rgb(original, bg))
            if added > args.fill_limit:
                print(f"  BỎ QUA --fill-holes: tô đặc sẽ thêm {added:.0f}% mực in đè lên áo cùng màu "
                      f"(ngưỡng {args.fill_limit:g}%). Cờ này dành cho ảnh có người, không dành cho poster.")
                silhouette = None
            else:
                keyed = filled
                filled_in = True
        cut = crop_to_content(keyed, min_alpha=40)
        shirt = {"black": (20, 20, 22), "white": (245, 245, 245)}.get(kind) or bg_color(original)
    cut.save(WORK_DIR / f"{src.stem}-cut.png")

    style = detect_style(cut) if args.style == "auto" else args.style
    colors = args.colors if args.colors is not None else (12 if args.vector else 0)
    model = "realesrgan-x4plus-anime" if style == "flat" else "realesrgan-x4plus"
    how = {"none": "đã trong suốt", "ai": "cắt hình" + (" + tinh chỉnh viền" if refine else ""),
           "black": "key nền đen", "white": "key nền trắng",
           "color": "key khoảng cách màu" if kind in ("black", "white") else "key màu nền"}[bg]
    how += (" + thân hình đặc" if bg != "ai" else " + lấp lỗ") if filled_in else ""
    shirt_txt = "" if bg == "none" else f" | áo: {SHIRT_LABEL['same' if bg != 'ai' else 'other']}"
    place_txt = "" if args.place == "center" and args.scale == 100.0 else f" | đặt: {args.place} {args.scale:g}%"
    print(f"  nền: {kind}{shirt_txt} | cách: {how} | kiểu: {style} | {'vector' if args.vector else 'raster'}{place_txt}"
          f"{f', gom {colors} màu' if colors else ', giữ nguyên màu'}")

    if args.vector:
        q = quantize(cut, colors, binary_alpha=True, merge_delta_e=args.merge)
        q_path = WORK_DIR / f"{src.stem}-quant.png"
        q.save(q_path)
        svg_path = WORK_DIR / f"{src.stem}.svg"
        trace_svg(q_path, svg_path)
        result = render_svg(svg_path, inner)
    elif bg in ("ai", "none"):
        big = upscale(cut, model=model)
        big = big.resize(fit_box(*big.size, inner), Image.Resampling.LANCZOS)
        result = flatten_raster(big, colors, args.merge) if colors else tighten_alpha(big)
    else:
        # keyed background: upscale the flat RGB first (cleaner edges, denoised background),
        # key at full resolution, then crop
        big_rgb = upscale(original.convert("RGB"), model=model)
        keyed = key_bg(big_rgb, bg, args.floor)
        if silhouette:
            keyed = solid_core(keyed, big_rgb, silhouette, bg_rgb(big_rgb, bg))
        keyed = crop_to_content(keyed, min_alpha=40)
        result = keyed.resize(fit_box(*keyed.size, inner), Image.Resampling.LANCZOS)
        if colors:
            result = flatten_raster(result, colors, args.merge)

    result = place_on_canvas(result, box, args.place, args.margin)
    name = out_name(src.stem, box, args.place, args.scale)
    out_path = OUTPUT_DIR / name
    save_print_png(result, out_path)
    make_review(original, result, REVIEW_DIR / name, shirt=shirt)
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
    daily.add_argument("--vector", action="store_true",
                       help="minh họa phẳng: gom màu rồi trace vector (mảng màu tuyệt đối phẳng, viền cong mượt). Mặc định là raster: upscale AI, giữ nguyên màu")
    daily.add_argument("--ui", action="store_true",
                       help="mở trang xem tại máy: chọn cờ theo từng ảnh, xem kết quả trên đúng màu áo, kèm số chấm chất lượng")
    daily.add_argument("files", nargs="*", help="chỉ xử lý các file này thay cho cả input/")
    p = p.add_argument_group("nâng cao (thường không cần)")
    p.add_argument("--bg", choices=["auto", "black", "white", "color", "ai", "none"], default="auto", help=argparse.SUPPRESS)
    p.add_argument("--style", choices=["auto", "flat", "detail"], default="auto",
                   help="chọn model upscale: flat = tranh phẳng, detail = tranh có gradient/texture. auto = tự nhận diện")
    p.add_argument("--colors", type=int, default=None,
                   help="gom về tối đa N màu. Mặc định: 12 khi --vector, không gom khi raster. 0 = không gom")
    p.add_argument("--merge", type=float, default=MERGE_DELTA_E,
                   help=f"ngưỡng gộp màu gần nhau (CIELAB ΔE, mặc định {MERGE_DELTA_E:g}). Tăng nếu còn đốm màu lệch, giảm nếu hai màu khác nhau bị gộp")
    p.add_argument("--fill-limit", type=float, default=FILL_LIMIT,
                   help=f"ngưỡng an toàn cho --fill-holes: nếu tô đặc làm tăng quá ngần này phần trăm mực in đè lên "
                        f"áo cùng màu thì bỏ qua và cảnh báo. Mặc định {FILL_LIMIT:g}. 100 = tắt chốt chặn")
    p.add_argument("--margin", type=float, default=2.0,
                   help="khoảng cách từ mép khung tới thiết kế khi --place không phải center, tính theo phần trăm cạnh ngắn. Mặc định 2")
    p.add_argument("--floor", type=float, default=KEY_FLOOR,
                   help=f"key màu nền: màu cách nền dưới ngưỡng này (khoảng cách RGB) coi như màu áo, cho trong suốt hẳn. "
                        f"Mặc định {KEY_FLOOR:g}, dập quầng xám mà máy in vẫn phủ lót trắng. 0 = tắt")
    p.add_argument("--fill-holes", action="store_true",
                   help="không đục lỗ trong hình. Cắt hình: lấp vùng trong suốt bị bao kín (chấm sáng, răng bị model khoét). "
                        "Key (--shirt same): thân hình theo model cắt hình được giữ đặc (bóng áo tối, da trùng màu nền), ngoài thân hình vẫn key. "
                        "Không dùng nếu thiết kế có lỗ xuyên cố ý")
    p.add_argument("--keep-input", action="store_true", help="không chuyển ảnh gốc sang input/done/")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.ui:
        import ui  # noqa: PLC0415 - chỉ nạp khi cần, để chạy dòng lệnh không phải nạp thêm gì

        ui.serve()
        return 0
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
