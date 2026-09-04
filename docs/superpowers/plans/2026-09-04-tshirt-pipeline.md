# tshirt-pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One command turns AI-generated flat illustrations in `input/` into print-ready transparent PNGs (300 DPI, fit in 30×40 cm) in `output/`, with side-by-side review images.

**Architecture:** A single `pipeline.py` with small pure functions (geometry, color, trace, render, upscale, background removal, export) composed by `process_one()`, driven by `main()` which iterates `input/`, isolates per-file errors, and moves sources to `input/done/` or `input/failed/`. Default mode traces to SVG via vtracer and renders via resvg; `--raster` mode upscales via realesrgan-ncnn-vulkan (Lanczos fallback) and quantizes.

**Tech Stack:** Python 3.12 via uv, rembg (birefnet-general, onnxruntime CPU), vtracer 0.6, resvg CLI (brew), Pillow, numpy, pytest.

## Global Constraints

- Python `>=3.11,<3.13`, env managed by `uv sync` (already done; `pyproject.toml` exists).
- Print DPI fixed at 300. Pixel = round(cm / 2.54 × 300). Default `--size 30x40` → (3543, 4724).
- Default `--colors 12`. rembg model `birefnet-general`, `alpha_matting=False`.
- vtracer params: `colormode="color", hierarchical="stacked", mode="spline", filter_speckle=4, color_precision=8, corner_threshold=60, path_precision=3`.
- Content crop margin 2 % per side. Review image height 800 px on checkerboard.
- Never touch system Python. All commands run via `uv run`.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

## File Structure

| File | Responsibility |
|---|---|
| `pipeline.py` | All logic. Sections: constants, geometry, color, vector, raster, remove_bg, export, process_one, CLI. |
| `run.sh` | `exec uv run python pipeline.py "$@"` from project dir. |
| `setup.sh` | brew deps, `uv sync`, download Real-ESRGAN to `bin/`, mkdir folders, `./run.sh --help`. |
| `tests/conftest.py` | Fixtures: synthetic RGBA images, tmp project dirs. |
| `tests/test_geometry.py` | `target_box_px`, `fit_box`, `crop_to_content`. |
| `tests/test_color.py` | `quantize`. |
| `tests/test_vector.py` | `trace_svg` + `render_svg` end-to-end (skip without resvg). |
| `tests/test_raster.py` | `upscale` Lanczos fallback, `flatten_raster`. |
| `tests/test_export.py` | `save_print_png`, `make_review`. |
| `tests/test_cli.py` | `main` batch behaviour with `remove_bg` monkeypatched. |
| `tests/test_rembg_slow.py` | `@pytest.mark.slow` real BiRefNet run. |
| `.gitignore` | `.venv/ input/ output/ review/ work/ bin/ __pycache__/ .pytest_cache/` |

---

### Task 1: Scaffold, run.sh, CLI skeleton

**Files:** Create `pipeline.py`, `run.sh`, `.gitignore`, `tests/__init__.py`, `tests/conftest.py`, `tests/test_cli.py`.

**Produces:** `pipeline.parse_args(argv) -> argparse.Namespace` with fields `raster: bool, colors: int, size: str, keep_input: bool, files: list[str]`; constants `ROOT, INPUT_DIR, OUTPUT_DIR, REVIEW_DIR, WORK_DIR, BIN_DIR, DPI=300`.

- [ ] **Step 1: Write failing test** `tests/test_cli.py`

```python
import pipeline

def test_parse_args_defaults():
    a = pipeline.parse_args([])
    assert a.raster is False and a.colors == 12 and a.size == "30x40"
    assert a.keep_input is False and a.files == []

def test_parse_args_flags():
    a = pipeline.parse_args(["--raster", "--colors", "6", "--size", "21x29.7", "--keep-input", "a.png"])
    assert a.raster and a.colors == 6 and a.size == "21x29.7" and a.keep_input and a.files == ["a.png"]
```

- [ ] **Step 2: Run** `uv run pytest tests/test_cli.py -v` → FAIL (no module `pipeline`).
- [ ] **Step 3: Implement** `pipeline.py` header + `parse_args`, `run.sh`, `.gitignore`, empty `tests/__init__.py`, `tests/conftest.py` adding project root to `sys.path`.
- [ ] **Step 4: Run** → PASS. Also `chmod +x run.sh && ./run.sh --help` prints usage.
- [ ] **Step 5: Commit** `feat: scaffold pipeline CLI`

### Task 2: Geometry

**Produces:** `target_box_px(size: str) -> tuple[int,int]`, `fit_box(w:int,h:int,box:tuple[int,int]) -> tuple[int,int]`, `crop_to_content(img: Image, margin: float = 0.02) -> Image` (raises `EmptyResult` if no alpha>0).

- [ ] Test `tests/test_geometry.py`: `target_box_px("30x40") == (3543, 4724)`, `("21x29.7") == (2480, 3508)`; `fit_box(1000,500,(3543,4724)) == (3543,1771)`; `fit_box(500,1000,(3543,4724)) == (2362,4724)`; crop of 400×400 transparent image with red square at (100,100)-(199,199) → size (104,104) and center pixel red; crop of fully transparent → `EmptyResult`.
- [ ] Run → FAIL. Implement. Run → PASS. Commit `feat: geometry helpers`.

### Task 3: Color quantization

**Produces:** `quantize(img: Image, colors: int, binary_alpha: bool) -> Image` (RGBA). RGB via `Image.quantize(colors, method=Image.Quantize.MEDIANCUT)` on an RGB copy where transparent pixels are first filled with the mean opaque color (so they don't steal palette slots), then alpha re-attached: binary threshold 128 if `binary_alpha`, else clean `<16→0`, `>240→255`.

- [ ] Test `tests/test_color.py`: image with 50 shades of red + transparent border, `quantize(img, 2, True)` → `len(set(opaque rgb)) <= 2`, alpha values ⊆ {0,255}; `binary_alpha=False` with alpha 10 → 0, alpha 250 → 255, alpha 128 → 128.
- [ ] Run → FAIL. Implement. Run → PASS. Commit `feat: quantize`.

### Task 4: Vector path

**Produces:** `trace_svg(png_path: Path, svg_path: Path) -> None`, `render_svg(svg_path: Path, box: tuple[int,int]) -> Image` (reads SVG width/height attrs via regex, `fit_box`, calls `resvg -w W svg out.png` in a temp file, returns RGBA), `check_tools() -> None` raising `SystemExit` with setup hint if `resvg` missing.

- [ ] Test `tests/test_vector.py` (`pytest.importorskip` not needed; `skipif shutil.which("resvg") is None`): red circle 400×400 on transparent → trace → render into (2000, 3000) → size (2000,2000), corner alpha 0, center (220,30,30,255).
- [ ] Run → FAIL. Implement. Run → PASS. Commit `feat: vector trace and render`.

### Task 5: Raster path

**Produces:** `upscale(img: Image, scale: int = 4) -> Image` (uses `bin/realesrgan-ncnn-vulkan -i in.png -o out.png -n realesrgan-x4plus-anime -s 4` if binary exists, else prints warning and `img.resize(LANCZOS)`), `flatten_raster(img: Image, colors: int) -> Image` (MedianFilter(5) on RGB, then `quantize(img, colors, binary_alpha=False)`).

- [ ] Test `tests/test_raster.py`: monkeypatch `pipeline.REALESRGAN_BIN` to a non-existent path → `upscale(100×50)` → (400,200); `flatten_raster` on noisy 2-color image with `colors=2` → ≤2 opaque colors.
- [ ] Run → FAIL. Implement. Run → PASS. Commit `feat: raster upscale and flatten`.

### Task 6: Background removal

**Produces:** `remove_bg(img: Image) -> Image` with module-level lazy `_SESSION = None`, `rembg.new_session("birefnet-general")`, `rembg.remove(img, session=_SESSION, alpha_matting=False)`.

- [ ] Test `tests/test_rembg_slow.py` marked `slow`: red circle on white 512×512 → result corner alpha 0, center alpha 255.
- [ ] Implement. Run `uv run pytest -m slow -v` once (downloads ~900 MB). Commit `feat: rembg wrapper`.

### Task 7: Export and review

**Produces:** `save_print_png(img: Image, path: Path) -> None` (saves with `dpi=(300,300)`), `make_review(original: Image, result: Image, path: Path) -> None` (both scaled to height 800, result over 32-px checkerboard, 20-px gap, saved as PNG).

- [ ] Test `tests/test_export.py`: saved PNG reopened has `info["dpi"] ≈ (300,300)`; review image height 800, width == w1 + 20 + w2.
- [ ] Run → FAIL. Implement. Run → PASS. Commit `feat: export and review`.

### Task 8: process_one and main

**Produces:** `process_one(src: Path, args) -> Path` (steps per spec §5, writes `work/<stem>-cut.png`, `work/<stem>.svg`, `output/<stem>.png`, `review/<stem>.png`), `main(argv=None) -> int` (check_tools, collect files `*.png *.jpg *.jpeg *.webp` from `INPUT_DIR` unless `args.files`, loop with try/except, move to `done/` or `failed/`, print summary table, return 1 if any failed).

- [ ] Test in `tests/test_cli.py`: fixture redirects `pipeline.INPUT_DIR/OUTPUT_DIR/REVIEW_DIR/WORK_DIR` into `tmp_path`, monkeypatch `pipeline.remove_bg` to identity, `pipeline.check_tools` to no-op; put `ok.png` (red circle on transparent) and `bad.png` (text file) into input; `main([])` returns 1; `output/ok.png` exists with DPI 300; `input/done/ok.png` and `input/failed/bad.png` exist; `review/ok.png` exists. Skip if no resvg.
- [ ] Run → FAIL. Implement. Run → PASS. Commit `feat: batch processing`.

### Task 9: setup.sh and real end-to-end

- [ ] Write `setup.sh`: `set -e; cd "$(dirname "$0")"; brew install uv resvg; uv sync; mkdir -p input/done input/failed output review work bin; curl -L <Real-ESRGAN macOS zip> -o bin/re.zip && unzip -o bin/re.zip -d bin && chmod +x bin/realesrgan-ncnn-vulkan && rm bin/re.zip || echo "WARN: Real-ESRGAN not installed; --raster falls back to Lanczos"; ./run.sh --help`.
- [ ] Run `./setup.sh`. Expected: finishes, prints usage.
- [ ] Generate a realistic test image (flat illustration with white background, drawn with Pillow: several shapes + text) into `input/`, run `./run.sh`, then `./run.sh --raster --keep-input` on a copy. Inspect `review/*.png` visually. Expected: transparent background, output 4724 px tall or 3543 px wide, DPI 300.
- [ ] Commit `feat: setup script`.

## Self-review

- Spec §4 flags → Task 1 + 8. §5.1 → Task 2, 6, 8. §5.2 → Task 3, 4. §5.3 → Task 5. §5.4 → Task 7, 8. §6 errors → Task 2 (`EmptyResult`), 4 (`check_tools`), 8 (batch isolation). §7 → Task 9. §8 tests → each task. No placeholders. Names consistent: `fit_box`, `crop_to_content`, `quantize`, `trace_svg`, `render_svg`, `upscale`, `flatten_raster`, `remove_bg`, `save_print_png`, `make_review`, `process_one`, `main`, `parse_args`, `check_tools`, `target_box_px`.
