# tshirt-pipeline

Biến ảnh minh họa phẳng do AI sinh (nền trơn, khoảng 1024–2048 px) thành file PNG nền trong suốt,
300 DPI, đúng kích thước file in (mặc định 4500 × 5100 px), sẵn sàng gửi xưởng in DTF/DTG.

Sửa ba lỗi thường gặp của ảnh AI khi in: mảng màu không đều, pixel lỗi khi phóng to, viền răng cưa.
Thay bước tách nền tay trong Photoshop.

## Cài một lần

```bash
./setup.sh
```

Cần Homebrew. Script cài `uv` và `resvg`, tạo môi trường Python riêng (không đụng Python hệ thống),
tải Real-ESRGAN cho chế độ raster. Lần chạy đầu tiên sẽ tải thêm model tách nền BiRefNet (~900 MB)
về `~/.rembg/models/`.

## Dùng hằng ngày

1. Thả ảnh (png, jpg, webp) vào `input/`.
2. Chạy:

   ```bash
   ./run.sh
   ```

3. Lấy kết quả:
   - `output/<tên>.png`: file in, nền trong suốt, 4500 × 5100 px, 300 DPI, thiết kế căn giữa.
   - `review/<tên>.png`: ảnh gốc bên trái, kết quả bên phải trên nền ca-rô. Lướt nhanh bằng Preview để phát hiện hình bị hỏng.
   - Ảnh gốc đã xử lý được chuyển sang `input/done/`, ảnh lỗi sang `input/failed/`.

Thử ngay với ảnh mẫu:

```bash
cp examples/mascot.jpg input/ && ./run.sh --fill-holes
```

## Các tùy chọn

| Cờ | Mặc định | Khi nào dùng |
|---|---|---|
| `--raster` | tắt | Khi kết quả vector làm hỏng chi tiết (nét quá mảnh, chữ nhỏ, chuyển sắc). Dùng upscale AI + gom màu thay cho trace vector. |
| `--colors N` | 12 | Số màu tối đa. Giảm (6–8) cho thiết kế ít màu để sạch hơn, tăng (16–24) cho thiết kế nhiều màu. |
| `--merge D` | 12 | Ngưỡng gộp hai màu gần nhau (CIELAB ΔE). Tăng nếu còn đốm màu lệch trong mảng phẳng, giảm nếu hai màu khác nhau bị gộp thành một. |
| `--fill-holes` | tắt | Khi tách nền khoét mất chi tiết trắng bên trong hình (chấm sáng mắt, răng). Không dùng nếu thiết kế có lỗ xuyên cố ý (lỗ chữ A, O, vòng tròn). |
| `--size WxH` | 4500x5100 | Kích thước file in. Số ≥ 200 hiểu là pixel, số nhỏ hơn hiểu là cm (`30x40` → 3543 × 4724 px). |
| `--keep-input` | tắt | Không chuyển ảnh gốc khỏi `input/`. |
| `file1 file2 …` | | Chỉ xử lý các file này thay cho cả `input/`. |

Ví dụ:

```bash
./run.sh --colors 8 --fill-holes            # thiết kế ít màu, có chấm sáng trắng
./run.sh --raster input/done/abc.png         # chạy lại một ảnh ở chế độ raster
./run.sh --size 30x40                        # khổ 30 x 40 cm thay cho 4500 x 5100 px
```

## Cách hoạt động

Mỗi ảnh đi qua:

1. **Tách nền** bằng rembg với model BiRefNet-general.
2. **Cắt** về khung bao của hình, chừa lề 2 %.
3. **Chế độ vector (mặc định):** gom màu → trace thành SVG bằng vtracer → vẽ lại bằng resvg ở đúng kích thước in.
   Mảng màu thành khối màu duy nhất, viền là đường cong nên mượt tuyệt đối.
4. **Chế độ raster (`--raster`):** upscale 4 lần bằng Real-ESRGAN (model anime) → co về khung in → siết viền alpha → gom màu.
5. **Xuất** PNG 300 DPI căn giữa trên canvas đúng kích thước, kèm ảnh so sánh.

Bước gom màu được thiết kế riêng cho ảnh AI: lọc nhiễu hạt, lấy palette từ pixel bên trong mảng
(pixel viền chỉ có trọng số thấp) nên chấm sáng nhỏ và viền mảnh vẫn giữ được màu; màu nào vừa
phân tán thành đốm nhỏ vừa gần một màu lớn thì được coi là nhiễu và hút vào màu đó; pixel rìa bán
trong suốt lấy màu của mảng kề bên để không tạo quầng sáng khi in trên áo tối.

## Mẹo khi sinh ảnh bằng AI

- Sinh thiết kế phẳng riêng trên nền trơn, không sinh trực tiếp lên mockup áo rồi cắt ra.
- Prompt gợi ý: `flat vector illustration, solid colors, clean sharp edges, no texture, no gradient, isolated on plain white background`.
- Nếu thiết kế có nhiều chi tiết trắng, hãy sinh trên nền màu khác (xanh lá, hồng) để bước tách nền không khoét mất chúng, hoặc dùng `--fill-holes`.
- Xuất PNG từ công cụ AI thay cho JPEG khi có thể.

## Kiểm thử

```bash
uv run pytest                 # test nhanh, ảnh tổng hợp
uv run pytest -m slow         # test tích hợp với model tách nền thật
```

## Cấu trúc

```
pipeline.py     toàn bộ logic
run.sh          ./run.sh [cờ] [file...]
setup.sh        cài một lần
examples/       ảnh mẫu để thử
tests/          pytest
docs/superpowers/   spec và kế hoạch triển khai
```
