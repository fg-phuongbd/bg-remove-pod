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

## Trang xem tại máy: `./run.sh --ui`

```bash
./run.sh --ui
```

Mở một trang tại `127.0.0.1:8765`, chạy bằng thư viện có sẵn của Python, không thêm phụ thuộc nào.
Trang làm ba việc mà dòng lệnh làm không tốt:

**Xem trên đúng màu áo sẽ in.** Ô chọn nền có bốn lựa chọn: màu áo, xám, checkerboard, trắng. Mặc
định là màu áo. File cho áo tối phần lớn là mực sáng, nên xem trên nền trắng hay checkerboard sáng
thì nét trắng biến mất và trông như thủng trong khi không hề thủng.

**Cờ nhớ theo từng ảnh.** Ảnh có người bật `thân hình đặc`, poster thì không, bản in nhỏ đặt vị trí
riêng. Mỗi thẻ ảnh hiện cờ của nó, và trình duyệt nhớ lại khi bạn mở trang lần sau.

**Ba con số chấm chất lượng** hiện ngay dưới ảnh:

| Số | Nghĩa |
|---|---|
| mực đặc | phần trăm pixel đục hoàn toàn. Thấp nghĩa là mực mỏng, in ra vải lộ qua. |
| mực trùng màu áo | phần trăm mực đục nhưng cùng màu áo, tức chỗ máy phủ lót trắng rồi in đè lên vải. Càng thấp càng tốt. |
| sai số khi in | ghép file lên màu áo rồi so với ảnh gốc, tính theo mức trên 255. |

Ảnh gốc được giữ nguyên tại chỗ, không bị chuyển sang `input/done/`, để chạy lại cùng một ảnh với cờ
khác được nhiều lần.

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
| `--fill-holes` | tắt | Không đục lỗ trong hình. Cắt hình: lấp chi tiết trắng bị model khoét (mắt, răng). Key `--shirt same`: thân hình nhân vật theo model cắt hình được giữ đặc (bóng áo tối trên nền đen, da trùng màu nền), ngoài thân hình vẫn key nên chữ ký, glow giữ nguyên. Không dùng nếu có lỗ chữ cố ý. |
| `--keep-input` | tắt | Không chuyển ảnh gốc khỏi `input/`. |
| `--floor D` | 32 | Key màu nền: màu cách nền dưới ngưỡng này (khoảng cách RGB) cho trong suốt hẳn. Dập quầng xám mà máy in vẫn phủ lót trắng. `0` = tắt. |
| `--bg X` | auto | Cờ ẩn, ép thẳng cách xử lý khi nhận diện nền sai: `black`, `white`, `color`, `ai`, `none`. Thắng `--shirt`. |

Ví dụ:

```bash
./run.sh                                   # áo cùng màu nền ảnh: nền đen in áo đen, nền hồng in áo hồng
./run.sh --shirt other                     # áo khác màu nền: cắt hình bằng model
./run.sh --shirt other input/done/abc.png  # chạy lại một ảnh nền đen để in lên áo trắng
./run.sh --fill-holes                      # ảnh có nhân vật: giữ thân hình đặc, không đục lỗ
./run.sh --vector --fill-holes             # minh họa phẳng nền trắng, có chấm sáng trắng
./run.sh --size 30x40                      # khổ 30 x 40 cm
```

## Nhân vật trên nền cùng màu áo: `--fill-holes`

Key nền cho alpha theo độ sáng, nên vùng thiết kế trùng hoặc gần màu nền sẽ thành trong suốt: da
người trên nền hồng, nếp áo tối trên nền đen. In lên áo đúng màu thì vẫn ra đúng ảnh gốc, nhưng file
nhìn như bị đục lỗ và chỗ đó in rất mỏng.

`--fill-holes` chạy thêm model cắt hình để lấy **thân hình**, rồi lấy alpha của model làm sàn: trong
thân hình mọi pixel đặc, kể cả bóng tối và da; ngoài thân hình vẫn key nguyên vẹn nên chữ ký, tia
sáng, glow không đổi. Màu được giải lại theo alpha mới, nên in lên áo đúng màu vẫn ra đúng ảnh gốc.

Chỉ hai thứ trong thân hình không được tô đặc, cả hai đều là nền mà key đã bỏ hẳn. Một là **viền**:
nền ngay ngoài thân hình, nới thêm 0,1% cạnh ngắn, giữ cho mép răng cưa dùng alpha mềm của chính nó
và nuốt luôn phần mask model lẹm ra ngoài. Hai là **khối thông ra mép khung**, tức chỗ ảnh đã mờ hết
vào nền mà model vẫn kéo thân người tới sát mép. Ngoài hai thứ đó, mọi chỗ model nhận là người đều
được tô đặc, kể cả cánh tay chìm hẳn vào bóng tối: nối liền với nền bên ngoài không có nghĩa là nền,
vì trên áo cùng màu thì bóng tối và áo vốn là cùng một pixel.

Sợi mask mảnh mà model vẽ bám theo nét sáng nhỏ (nét chữ ký, sợi tóc bay) bị loại trước bằng phép mở
hình thái học, vì tô đặc sợi đó sẽ biến khoảng nền kẹt bên trong thành mực đen đục.

Đổi lại: lần chạy đầu phải tải model cắt hình (~900 MB) và mỗi ảnh chậm thêm khoảng 20 giây. Không
dùng cờ này nếu thiết kế có lỗ xuyên cố ý.

## Hai kiểu key, pipeline tự chọn

Key nền đen hay trắng cho alpha bằng **độ sáng**. Đó là thứ tranh có glow, airbrush hay ảnh chụp cần:
chỗ tối mỏng dần rồi tan vào áo. Với **đồ họa phẳng** màu khối (chữ, logo, huy hiệu) thì sai: mảng đỏ
`(215, 8, 22)` chỉ ra alpha 214, mực đục 84% và vải lộ qua nét chữ.

Nên pipeline đo ảnh gốc trước khi key. Nếu gần như mọi pixel mực nằm trong mảng màu đều thì đó là đồ
họa phẳng, và nền được key theo **khoảng cách màu** thay vì độ sáng, cho màu khối đặc hoàn toàn. Còn
lại giữ key độ sáng.

Ngay trong key độ sáng cũng có một lớp nữa: **mảng màu khối được nâng lên đặc**. Với mỗi pixel,
pipeline hỏi bao nhiêu phần lân cận có cùng màu với nó. Mảng mực khối trả lời gần hết cửa sổ, dốc
glow chỉ trả lời đúng dải mỏng của bậc nó đang đứng, còn bóng đổ trong ảnh chụp thì ít hơn nữa. Trên
một thiết kế pha trộn, ba con số đó là 0,64, 0,17 và 0,31. Chỗ nào vượt ngưỡng thì alpha được **nhân hệ
số**, không phải gán cứng bằng 255. Alpha ở đó vốn là độ phủ nhân với độ sáng của chính màu đó, nên
chia ngược độ sáng ra sẽ đưa ruột mảng lên đặc và kéo cả dải răng cưa ở mép lên cùng hệ số, giữ mép
vẫn là mép. Gán cứng thì một pixel mép đang phủ 9% sẽ nhảy thẳng lên đặc, cho ra viền cứng, phình và
lốm đốm.

Nhờ vậy thiết kế **pha trộn**, tức ảnh chụp cộng chữ đồ họa, không cần cờ nào: chữ đỏ `(195, 20, 25)`
ra đặc thay vì 76%, còn glow và bóng đổ vẫn tan vào áo. Việc nâng alpha không bao giờ đổi kết quả in,
vì màu được giải lại theo alpha mới; nó cũng không tạo mực ở chỗ vốn trong suốt. Đo trên 10 ảnh nền
đen: tỉ lệ pixel đặc từ 21,2% lên 30,9%, sai số khi in giảm từ 1,10 xuống 1,04 mức trên 255, không
ảnh nào thêm một pixel mực nào. Dòng log in ra `cách: key khoảng cách màu` hay `cách: key nền đen` để bạn biết
nó chọn gì. Ép tay bằng `--bg black` hoặc `--bg color` nếu nhận diện sai.

Phép đo này khắt khe hơn `kiểu: flat/detail` ở cùng dòng log, vì cái đó chỉ dùng để chọn model
upscale. Nó cũng đo trên ảnh gốc chứ không đo trên bản đã key: key độ sáng làm mực tối nhạt đi, nên
trên ảnh chụp nó chỉ còn lại các mảng sáng mịn và trông như phẳng.

Đi kèm key khoảng cách là `--floor`, mặc định 32. Ảnh AI hay có mảng *gần* đen chứ không đen tuyệt đối, ví dụ
một hình ellipse `(15, 13, 13)` sau logo. Key khoảng cách sẽ cho nó alpha khoảng 80 trên 255. Ghép
trên màn hình nền đen thì vẫn đúng, nhưng máy DTG/DTF phủ lớp lót trắng theo alpha, nên mảng đó in ra
thành vệt xám nổi trên vải. Ngưỡng sàn đẩy mọi màu cách nền dưới 32 về trong suốt hẳn, cắt sau khi đã
tính xong dốc alpha nên mép chữ và màu viền không đổi. Trên thiết kế mẫu, quầng xám giảm từ 8% diện
tích xuống 0,5%. Đổi lại, chi tiết gần đen thật sự cũng mất, lệch nhiều nhất 31 mức trên 255, mức này
mắt không thấy trên vải đen. Đặt `--floor 0` để tắt.

## Đặt hình nhỏ trên khung in: `--place` và `--scale`

Mặc định thiết kế được phóng lấp đầy khung in rồi căn giữa. Với bản in nhỏ, ví dụ một hình ở góc
trên lưng áo hay in ngực trái, dùng hai cờ này:

```bash
./run.sh --place top-right --scale 26
./run.sh --place center --scale 40 --size 30x40
```

`--scale` là phần trăm của **cả khung**, không phải của chiều rộng, nên con số có ý nghĩa như nhau
với hình ngang hay hình dọc: 26% của khung 4500 x 5100 là hộp 1170 x 1326, hình ngang dùng hết chiều
rộng hộp đó còn hình dọc dùng hết chiều cao.

`--margin` là khoảng hở từ mép khung, mặc định 2% cạnh ngắn, chỉ có tác dụng khi `--place` khác
`center`. Hình căn giữa thì không bao giờ chạm tới nó.

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
