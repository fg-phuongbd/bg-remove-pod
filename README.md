# tshirt-pipeline

Biến ảnh thiết kế do AI sinh (khoảng 1024–2048 px, nền đen, trắng hoặc trơn) thành file PNG nền
trong suốt, 300 DPI, đúng kích thước file in (mặc định 4500 × 5100 px), sẵn sàng gửi xưởng in DTF/DTG.

Sửa các lỗi thường gặp của ảnh AI khi in: pixel lỗi khi phóng to, viền răng cưa, và với minh họa phẳng
là mảng màu không đều. Thay bước tách nền tay trong Photoshop.

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

Thử ngay với ảnh mẫu (minh họa phẳng nền trắng):

```bash
cp examples/mascot.jpg input/ && ./run.sh --vector --fill-holes
```

## Một câu hỏi duy nhất: in lên áo màu gì?

Pipeline nhìn **nền** của ảnh gốc (vành 2 % quanh ảnh: đen, trắng, màu trơn, hay đã trong suốt) và
hỏi một điều: áo in **cùng màu nền** hay **khác màu nền**. Cờ `--shirt` trả lời câu đó.

**Mặc định là `same`: sinh ảnh AI trên nền đúng màu áo sẽ in** (áo đen thì nền đen, áo hồng thì nền
hồng). Khi đó nền được tách bằng công thức, chính xác từng pixel, không cần model và không có vụn.
Chỉ dùng `--shirt other` khi buộc phải in một thiết kế lên áo khác màu nền gốc; đường này dùng model
và có thể để lại vụn nhỏ ở viền.

| Nền ảnh gốc | `--shirt same` (mặc định, áo cùng màu nền) | `--shirt other` (áo khác màu) | `--shirt auto` chọn |
|---|---|---|---|
| **đen** | Key nền đen: pixel đen thành trong suốt để áo hiện ra, màu được bù để in lên áo đen ra đúng ảnh gốc. Glow, airbrush giữ nguyên. | Cắt hình bằng BiRefNet, viền mềm được khử màu nền dính (Decontaminate Colors) | `same` (tranh nền đen là cho áo tối) |
| **trắng** | Key nền trắng, cho thiết kế có glow trắng in áo trắng | Cắt hình bằng BiRefNet **+ tinh chỉnh viền** (như nền màu). Chi tiết trắng nhỏ bên trong mà model khoét nhầm được lấp lại; vùng trắng lớn bên trong nét viền vẫn trong suốt. | `other` |
| **màu trơn** (hồng, xanh…) | Key màu nền: alpha theo khoảng cách tới màu nền, bù màu. Vùng vẽ bằng đúng màu nền cũng trong suốt. | Cắt hình bằng BiRefNet **+ tinh chỉnh viền**: bên trong tin model, dải viền quyết định từng pixel theo màu nền, nên bỏ được dải nền dính viền và lấy lại tia, chấm mực mảnh mà model cắt cụt. | `other` |
| **đã trong suốt** | Bỏ qua tách nền | Bỏ qua tách nền | |

Dòng đầu khi xử lý in ra quyết định này, ví dụ `nền: color | áo: khác màu nền | cách: cắt hình + tinh chỉnh viền`.

Sau đó ảnh đi qua **raster** (mặc định): upscale 4 lần bằng Real-ESRGAN, giữ nguyên màu và chi tiết,
co về khung in. Model upscale được chọn theo kiểu thiết kế: tranh phẳng và line art ít màu dùng model
anime, tranh có gradient/texture dùng model chung. Với **minh họa phẳng thật sự** (mảng màu, không gradient), thêm
`--vector` để gom màu rồi trace vector: mảng màu tuyệt đối phẳng, viền cong mượt ở mọi kích cỡ.

## Các tùy chọn

Hằng ngày chỉ cần ba cờ:

| Cờ | Mặc định | Khi nào dùng |
|---|---|---|
| `--shirt X` | same | `same` = áo cùng màu nền ảnh, `other` = áo khác màu, `auto` = nền đen coi là áo đen, còn lại áo khác màu. Xem bảng trên. |
| `--size WxH` | 4500x5100 | Kích thước file in. Số ≥ 200 là pixel, nhỏ hơn là cm (`30x40`). |
| `--vector` | tắt | Minh họa phẳng: gom 12 màu rồi trace vector. Không dùng cho tranh có gradient, texture, chữ rất nhỏ. |
| `file1 file2 …` | | Chỉ xử lý các file này. |

Nâng cao, thường không cần:

| Cờ | Mặc định | Khi nào dùng |
|---|---|---|
| `--style X` | auto | Ép model upscale: `flat` hoặc `detail`. |
| `--colors N` | không gom | Gom về N màu (raster) hoặc đổi số màu khi `--vector` (mặc định 12). |
| `--merge D` | 12 | Ngưỡng gộp hai màu gần nhau khi gom (CIELAB ΔE). Tăng nếu còn đốm màu lệch, giảm nếu hai màu khác bị gộp. |
| `--fill-holes` | tắt | Khi cắt hình: lấp chi tiết trắng bên trong hình bị model khoét mất (mắt, răng). Không dùng nếu có lỗ chữ cố ý. |
| `--keep-input` | tắt | Không chuyển ảnh gốc khỏi `input/`. |
| `--bg X` | auto | Cờ ẩn, ép thẳng cách xử lý khi nhận diện nền sai: `black`, `white`, `color`, `ai`, `none`. Thắng `--shirt`. |

Ví dụ:

```bash
./run.sh                                   # áo cùng màu nền ảnh: nền đen in áo đen, nền hồng in áo hồng
./run.sh --shirt other                     # áo khác màu nền: cắt hình bằng model
./run.sh --shirt other input/done/abc.png  # chạy lại một ảnh nền đen để in lên áo trắng
./run.sh --vector --fill-holes             # minh họa phẳng nền trắng, có chấm sáng trắng
./run.sh --size 30x40                      # khổ 30 x 40 cm
```

## Lưu ý về file in cho áo tối

Với ảnh nền đen, file ra có nhiều vùng **bán trong suốt** (glow, airbrush, vùng tối). Đây là cách
chuẩn để in DTF/DTG lên áo đen: máy in dùng lớp lót trắng theo độ trong suốt, áo đen đóng vai trò màu
đen của thiết kế. File này **chỉ đúng khi in lên áo đen hoặc rất tối**. In lên áo sáng thì dùng `--shirt other`.

## Cách hoạt động

1. **Nhận diện nền** qua viền ảnh, rồi tách nền theo bảng trên. Với nền đen/trắng, ảnh được upscale
   trước rồi mới tách để viền sạch và nền bớt nhiễu.
2. **Cắt** về khung bao của hình, chừa lề 2 %.
3. **Raster (mặc định):** upscale 4 lần bằng Real-ESRGAN, co về khung in, siết viền alpha nhẹ.
   **`--vector`:** gom màu → trace SVG bằng vtracer → vẽ lại bằng resvg đúng kích thước in.
4. **Xuất** PNG 300 DPI căn giữa trên canvas đúng kích thước, kèm ảnh so sánh trong `review/`
   (nền đen thì xem trên nền áo đen, còn lại xem trên nền ca-rô).

Bước gom màu (khi dùng) được thiết kế riêng cho ảnh AI: lọc nhiễu hạt, lấy palette từ pixel bên trong
mảng nên chấm sáng nhỏ và viền mảnh vẫn giữ màu; màu vừa phân tán thành đốm vừa gần một màu lớn
được coi là nhiễu; pixel rìa bán trong suốt lấy màu mảng kề để không tạo quầng khi in.

## Mẹo khi sinh ảnh bằng AI

- Sinh thiết kế riêng trên nền trơn, không sinh trực tiếp lên mockup áo rồi cắt ra.
- In áo đen: sinh trên nền đen thuần. In áo trắng: sinh trên nền trắng thuần. Nền càng đều thì tách càng sạch.
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
