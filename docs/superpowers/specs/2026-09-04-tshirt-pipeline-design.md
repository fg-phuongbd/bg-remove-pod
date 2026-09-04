# tshirt-pipeline — Thiết kế

Ngày: 2026-09-04. Trạng thái: đã duyệt và triển khai (xem §10 cho các thay đổi).

## 1. Mục tiêu

Biến ảnh minh họa phẳng do AI sinh (nền trơn, độ phân giải khoảng 1024–2048 px) thành
file PNG nền trong suốt, 300 DPI, vừa trong khung in 30 × 40 cm, sẵn sàng gửi xưởng in DTF/DTG.
Sửa ba lỗi: mảng màu không đều, pixel lỗi khi phóng to, viền răng cưa.
Thay thế bước tách nền tay trong Photoshop.

## 2. Phạm vi

Làm:
- Xử lý hàng loạt mọi ảnh trong `input/` bằng một lệnh.
- Hai chế độ dùng chung bước tách nền và xuất PNG:
  - **vector** (mặc định): trace thành vector rồi vẽ lại. Sửa tận gốc cả ba lỗi.
  - **raster** (cờ `--raster`): upscale AI + gom màu. Dùng khi trace làm hỏng chi tiết.
- Ảnh so sánh trước/sau để duyệt nhanh bằng mắt.

Không làm:
- Không tự tìm vùng thiết kế trong ảnh mockup nguyên áo. Đầu vào phải là hình phẳng trên nền trơn.
- Không xuất SVG cho người dùng (SVG chỉ là file trung gian, được giữ lại trong `work/` để gỡ lỗi).
- Không có giao diện đồ họa, không theo dõi thư mục tự động.

## 3. Cấu trúc thư mục

```
~/Phuong-data/tshirt-pipeline/
  setup.sh          # cài một lần
  run.sh            # gọi pipeline.py trong môi trường uv
  pipeline.py       # toàn bộ logic, một file
  pyproject.toml    # phụ thuộc Python (uv quản lý)
  input/            # người dùng thả ảnh vào đây (png, jpg, jpeg, webp)
  input/done/       # ảnh gốc đã xử lý thành công được chuyển vào đây
  input/failed/     # ảnh xử lý lỗi được chuyển vào đây
  output/           # PNG kết quả, cùng tên file gốc, đuôi .png
  review/           # ảnh so sánh gốc | kết quả, cùng tên file gốc
  work/             # file trung gian (đã tách nền, svg), có thể xóa
  bin/              # realesrgan-ncnn-vulkan + models/ (chỉ cho chế độ raster)
  tests/            # pytest, dùng ảnh tổng hợp tự vẽ
```

## 4. Giao diện dòng lệnh

```
./run.sh [--raster] [--colors N] [--size WxH] [--keep-input] [files...]
```

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--raster` | tắt | Dùng chế độ raster thay cho vector |
| `--colors N` | 12 | Số màu tối đa sau khi gom. Áp dụng cho cả hai chế độ |
| `--size WxH` | `30x40` | Khung in theo cm. Pixel = cm / 2.54 × 300, làm tròn |
| `--keep-input` | tắt | Không xóa ảnh gốc khỏi `input/` sau khi thành công. Mặc định: ảnh thành công được chuyển sang `input/done/` |
| `files...` | rỗng | Nếu có, chỉ xử lý các file này thay cho cả `input/` |

Kết thúc, script in bảng tóm tắt: số ảnh thành công, số ảnh lỗi kèm lý do từng ảnh.
Mã thoát 0 nếu không có ảnh lỗi, 1 nếu có.

## 5. Luồng xử lý mỗi ảnh

### 5.1 Chung: tách nền
1. Mở ảnh, chuyển RGBA.
2. Tách nền bằng `rembg` với model `birefnet-general`. Dùng `alpha_matting=False`
   (hình phẳng, viền cứng; matting làm viền mờ).
3. Cắt về khung bao của pixel có alpha > 0, chừa lề 2 % mỗi cạnh.
4. Lưu `work/<tên>-cut.png`.

### 5.2 Chế độ vector (mặc định)
5. Gom màu: Pillow `quantize(colors=N, method=MEDIANCUT)` trên kênh RGB.
   Alpha nhị phân hóa ở ngưỡng 128 (đường trace sẽ tự làm mượt viền).
6. Trace bằng `vtracer` (Python binding), chế độ `color`, `hierarchical=stacked`,
   `mode=spline`, `filter_speckle=4`, `color_precision=8` (giữ nguyên N màu đã gom),
   `corner_threshold=60`, `path_precision=3`. Lưu `work/<tên>.svg`.
7. Vẽ SVG thành PNG bằng `resvg` CLI. Tính tỉ lệ để hình vừa trong khung pixel mục tiêu,
   truyền `--width` hoặc `--height` theo cạnh chạm khung. Nền trong suốt.

### 5.3 Chế độ raster (`--raster`)
5. Upscale ×4 bằng `realesrgan-ncnn-vulkan`, model `realesrgan-x4plus-anime`
   (hợp tranh minh họa). Nếu binary không có, cảnh báo và dùng Lanczos của Pillow.
6. Làm phẳng nhiễu: `MedianFilter(size=5)` trên RGB.
7. Gom màu về N màu như 5.2 bước 5, nhưng alpha giữ nguyên độ mượt (không nhị phân hóa),
   chỉ làm sạch alpha < 16 về 0 và > 240 về 255.
8. Co/giãn về vừa khung pixel mục tiêu bằng Lanczos.

### 5.4 Chung: xuất
9. Ghi `output/<tên>.png` bằng Pillow với `dpi=(300, 300)`.
10. Ghi `review/<tên>.png`: gốc bên trái, kết quả bên phải, cao 800 px, nền ca-rô
    để thấy vùng trong suốt.
11. Chuyển ảnh gốc sang `input/done/` (trừ khi `--keep-input`).

## 6. Xử lý lỗi

- Mỗi ảnh chạy trong `try/except`. Lỗi được ghi lại, ảnh gốc chuyển sang `input/failed/`,
  lô tiếp tục.
- Ảnh sau tách nền mà không còn pixel nào có alpha > 0: báo lỗi "tách nền ra rỗng".
- File không đọc được bằng Pillow: báo lỗi "không phải ảnh".
- Thiếu `resvg` hoặc `vtracer`: dừng ngay từ đầu với hướng dẫn chạy `setup.sh`.
- Lần đầu chạy, rembg tải model BiRefNet (~900 MB) về `~/.u2net/`. Script in thông báo trước.

## 7. Cài đặt (`setup.sh`)

1. `brew install uv resvg`.
2. `uv sync` trong thư mục dự án: tạo `.venv` Python 3.11 với `rembg[cpu]`, `vtracer`,
   `pillow`, `onnxruntime`, `pytest`.
3. Tải `realesrgan-ncnn-vulkan` bản macOS từ GitHub releases vào `bin/`, kèm thư mục `models/`.
   Bước này được phép thất bại (chỉ ảnh hưởng chế độ raster).
4. Tạo các thư mục `input/ output/ review/ work/`.
5. Chạy `./run.sh --help` để xác nhận.

## 8. Kiểm thử

Pytest với ảnh tổng hợp tự vẽ bằng Pillow (không phụ thuộc model AI cho phần lớn test):
- `tính pixel khung`: `30x40` → `(3543, 4724)`; `21x29.7` → `(2480, 3508)`.
- `vừa khung giữ tỉ lệ`: ảnh 1000×500 vào khung 3543×4724 → cạnh rộng 3543, cao 1771.
- `cắt khung bao`: ảnh RGBA có hình vuông đỏ giữa nền trong suốt → cắt đúng khung + lề 2 %.
- `gom màu`: ảnh có 50 sắc đỏ gần nhau + `--colors 2` → còn đúng ≤ 2 màu.
- `vector end-to-end` (bỏ qua nếu thiếu resvg): hình tròn đỏ trên nền trong suốt →
  PNG ra đúng kích cỡ, DPI 300, tâm là màu đỏ, góc là trong suốt.
- `lỗi không dừng lô`: một file text giả trong input → chuyển sang `failed/`, các ảnh khác vẫn ra.
- Tách nền bằng rembg được test bằng một test tích hợp riêng, đánh dấu `slow`, chạy tay.

## 9. Quyết định kỹ thuật

- **Gom màu trước khi trace** thay cho việc để vtracer tự gom: cờ `--colors` mới có ý nghĩa
  chính xác, và kết quả nhất quán giữa hai chế độ.
- **Nhị phân hóa alpha trước khi trace**: viền bán trong suốt làm vtracer sinh nhiều mảnh
  rìa nhỏ. Đường spline sau trace đã mượt hơn viền gốc.
- **resvg thay cho cairosvg**: chất lượng anti-alias tốt hơn, cài bằng brew, không kéo cairo.
- **Một file `pipeline.py`**: dự án nhỏ, một người dùng. Tách hàm rõ ràng bên trong
  (`fit_box`, `crop_to_content`, `quantize`, `trace_svg`, `render_svg`, `remove_bg`,
  `process_one`, `main`) để test độc lập.

## 10. Thay đổi so với thiết kế ban đầu (chốt khi triển khai, 2026-09-04)

- **Kích thước mặc định** đổi từ khung 30 × 40 cm sang file in `4500x5100` px theo yêu cầu người dùng.
  `--size` nhận pixel (số ≥ 200) hoặc cm. Kết quả được **căn giữa trên canvas đúng kích thước** thay cho
  "vừa trong khung": xưởng in nhận file cố định 4500 × 5100.
- **Gom màu tự viết** thay cho `Image.quantize(MEDIANCUT)`: median cut của Pillow không dành slot cho
  vùng nhỏ (chấm sáng mắt bị đổi màu). Cách mới: bin 16 mức/kênh trên pixel bên trong mảng (pixel viền
  trọng số 0.1, pixel bán trong suốt trọng số 0), gộp agglomerative theo ΔE CIELAB, luôn gộp cặp gần
  hơn `--merge` (mặc định 12), hút màu "vừa phân tán vừa gần màu khác" (`_absorb_scattered`), và cho
  pixel rìa bán trong suốt lấy màu mảng kề (`_decontaminate_fringe`).
- **Cờ mới `--merge D`** và **`--fill-holes`** (lấp vùng trong suốt bị bao kín, vì BiRefNet khoét
  chi tiết trắng cùng màu nền). `--fill-holes` mặc định tắt vì phá lỗ chữ cố ý.
- **vtracer `filter_speckle`** tính theo ảnh: `round(sqrt(w*h/15000))` (vtracer bình phương giá trị
  này thành diện tích).
- **Chế độ raster**: co về khung in *trước* khi gom màu (co sau sẽ trộn màu lại), thêm bước siết
  alpha (`tighten_alpha`, 96→160) để viền in không bị quầng.
- Model rembg tải về `~/.rembg/models/`, không phải `~/.u2net/`.
