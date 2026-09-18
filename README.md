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

1. Thả ảnh (png, jpg, webp) vào `input/`, hoặc kéo thẳng vào trang `./run.sh --ui`.

2. Chọn cờ theo **loại thiết kế**. Đây là toàn bộ những gì cần nhớ:

   | Thiết kế | Lệnh |
   |---|---|
   | Đồ họa phẳng: chữ, logo, huy hiệu | `./run.sh` |
   | Poster vẽ tay, có glow hoặc halftone | `./run.sh` |
   | Ảnh chụp người hoặc nhân vật | `./run.sh --fill-holes` |
   | Ảnh chụp người kèm chữ đồ họa | `./run.sh --fill-holes` |
   | Logo ngực trái | `./run.sh --preset chest-left` |
   | Bản in nhỏ ở vị trí khác | `./run.sh --place top-right --scale 26` |

   Ngoài `--fill-holes` và vị trí in, pipeline tự quyết phần còn lại: nhận diện nền, chọn kiểu key,
   chọn model upscale. Dòng log in ra nó đã chọn gì, và mỗi file in **ghi lại bộ cờ đã dùng** trong
   chính nó, xem mục **File in nhớ cách nó được tạo**.

3. Lấy kết quả:

   - `output/<tên>_4500x5100_center.png`: file in, nền trong suốt, 300 DPI. Tên mang theo khung in
     và vị trí, xem mục **Tên file in**.
   - `review/<cùng tên>.png`: ảnh gốc bên trái, kết quả bên phải **ghép trên màu áo sẽ in**.
   - Ảnh gốc chuyển sang `input/done/`, ảnh lỗi sang `input/failed/`.

4. Chấm kết quả bằng `./run.sh --ui`, để ô **nền xem** ở **màu áo**.

   Đừng chấm trên nền trắng hay checkerboard sáng. Thiết kế cho áo tối phần lớn là mực sáng, nên nét
   trắng biến mất vào nền trắng và trông như thủng trong khi không hề thủng. Khoảng trống giữa các nét
   chữ vốn là nền, thấy checkerboard ở đó là đúng.

Thử ngay với ảnh mẫu (minh họa phẳng nền trắng):

```bash
cp examples/mascot.jpg input/ && ./run.sh --vector --fill-holes
```

## Trang chạy tại máy: `./run.sh --ui`

```bash
./run.sh --ui
```

Mở một trang tại `127.0.0.1:8765`, chạy bằng thư viện có sẵn của Python, không thêm phụ thuộc nào,
không gửi ảnh đi đâu.

### Năm bước

**1. Thêm ảnh.** Bấm **Thêm ảnh…** ở cột trái, kéo thả vào bất cứ đâu trên trang, hoặc chép vào
`input/` rồi tải lại trang. Nhận png, jpg, webp. Ảnh gốc được giữ nguyên tại chỗ, không bị chuyển sang `input/done/`, nên chạy lại
cùng một ảnh với cờ khác bao nhiêu lần cũng được.

**2. Chọn ảnh, xem cờ của nó.** Bấm vào thẻ bên trái. Thanh cờ trên cùng luôn hiện cờ *của riêng ảnh
đang chọn*, không phải cờ chung. Dưới tên mỗi thẻ ghi những cờ khác mặc định, ví dụ
`áo other · đặt top-right · gom màu 8`. Trình duyệt nhớ những cờ này cả khi bạn đóng trang.

**3. Đặt cờ theo loại thiết kế.** Hầu hết thời gian chỉ cần một cờ; phần còn lại pipeline tự quyết
và in ra dòng log cho biết nó chọn gì.

| Thiết kế | Đặt gì |
|---|---|
| Đồ họa phẳng: chữ, logo, huy hiệu | để nguyên mặc định |
| Poster vẽ tay, có glow hoặc halftone | để nguyên mặc định |
| Ảnh chụp người hoặc nhân vật | bật `thân hình đặc` |
| Ảnh chụp người kèm chữ đồ họa | bật `thân hình đặc` |
| Logo ngực trái | `mẫu vị trí` = chest-left |
| Bản in nhỏ ở vị trí khác | `đặt` = top-right, `cỡ` = 26 |
| In một màu mực | `mực` = đen hoặc trắng |

Chín cờ hay dùng nằm trên thanh trên cùng, tám cờ còn lại trong khối `cờ nâng cao`. Danh sách và giá
trị mặc định lấy thẳng từ `parse_args`, nên thêm cờ mới vào dòng lệnh là trang có ngay.

**4. Chạy.** *Chạy ảnh đang chọn* làm một ảnh, *Chạy tất cả đang chờ* làm hết những ảnh chưa xử lý,
mỗi ảnh dùng cờ riêng của nó. Ô `cùng lúc` đặt số ảnh chạy song song, mặc định 2; đẩy lên 3 thì nhanh
hơn rõ, cao hơn nữa thì máy đuối vì mỗi ảnh giữ vài mảng cỡ 5000 x 5000 trong bộ nhớ. Mỗi ảnh mất
khoảng 14 giây, hoặc 45 giây nếu bật `thân hình đặc` vì phải chạy thêm model cắt hình.

Hai khung ảnh luôn thu cả file vào cho vừa. Bấm vào ảnh để xem 1:1 bản xem trước 1400 px và cuộn
soi mép, bấm lần nữa để thu về.

**5. Đọc kết luận.** Ô màu trả lời thẳng, các con số bên cạnh là dẫn chứng. Dưới các con số là
`cách` pipeline đã chọn và `cờ đã dùng`, một dòng lệnh dán lại vào terminal là ra đúng file này.

Một ảnh gốc có nhiều file in (chạy `chest-left` rồi chạy `center`) thì ô **bản in** cạnh nút tải
hiện ra để chọn bản muốn xem; không chọn thì là bản mới nhất, tức file vừa chạy xong.

| Mức | Nghĩa |
|---|---|
| Đủ điều kiện in | không thấy vấn đề nào, tải file về và gửi xưởng |
| In được, nên xem lại | phủ thấp, nét mảnh hay đốm nhỏ vượt ngưỡng DTF, nhiều màu ngoài gamut, hoặc mực mỏng bất thường |
| Không dùng được | file rỗng, mực in đè lên áo cùng màu quá 20%, hoặc sai số khi in quá 5 |

Luật quyết định nằm ở `print_verdict` trong `pipeline.py`, dùng chung với `--audit`, nên dòng lệnh và
trang không bao giờ nói khác nhau.

### Các con số

| Số | Nghĩa | Đọc thế nào |
|---|---|---|
| mực đặc | phần trăm pixel đục hoàn toàn | So trong cùng loại thiết kế. Logo phẳng dưới 85% là đáng ngờ; poster halftone 42% vẫn đúng. |
| mực trùng màu áo | mực đục nhưng cùng màu áo, tức chỗ máy phủ lót trắng rồi in đè lên vải | Dưới 5% thì bỏ qua. Trên 20% gần như chắc là bật `thân hình đặc` nhầm cho poster. |
| phủ thấp | mực nằm dưới 40% độ phủ | Cột quan trọng nhất với **in DTF**: vùng đó nhận ít bột keo nên dễ bong. Trên 5% thì in thử một chiếc, giặt vài lần rồi hãy chạy số lượng. In DTG có lót trắng thì không sao. |
| nét mảnh | mực nằm trong nét mảnh hơn 0,5 mm | DTF bám kém ở nét mảnh. Trên 5% thì bật `nới nét DTF` (`--dtf-safe`) hoặc in thử. Poster halftone đo được 8 đến 16%, chữ và logo dưới 3%. |
| đốm nhỏ | mực là đốm rời nhỏ hơn 1 mm² | Đốm dễ rơi khỏi bàn ép. Trên 1% thì như trên. Đốm mờ đã được dọn sẵn khi lưu; đây là đốm đậm. |
| ngoài gamut | mực lệch quá 25 ΔE sau khi đi qua hồ sơ CMYK, tức xỉn hẳn | Xanh lá chói, xanh dương thuần, tím, đỏ 255 của ảnh AI không mực nào pha ra. Trên 30% thì bật **xem như in** để thấy trước và quyết định có đổi màu không. Xem mục **Màu in được và màu không**. |
| sai số khi in | ghép file lên màu áo rồi so với ảnh gốc, thang 0 đến 255 | Dưới 1 là mắt không thấy. Trên 5 thì mở ảnh so sánh trong `review/` xem bằng mắt. |

Hai cột sau chỉ có nghĩa khi áo là màu nền ảnh gốc. File cắt hình (`áo` = other) in lên áo khác màu
mà trang không biết là màu gì, nên hai cột đó hiện `—` thay cho một con số so nhầm áo; chọn màu áo ở
ô **nền xem** để chấm bằng mắt. File một màu mực cố ý khác ảnh gốc, nên riêng "sai số" là `—`.

Nút **Chấm cả lô** hiện bảng này cho mọi file in một lượt, tô vàng những file cần xem lại kèm lý do.
Bấm lần nữa để đóng. Ngoài dòng lệnh: `./run.sh --audit`.

### Ba cái bẫy

**Đừng chấm trên nền trắng.** Thiết kế cho áo tối phần lớn là mực sáng. Trên nền trắng hay
checkerboard sáng, nét trắng biến mất và trông như thủng lỗ trong khi không hề thủng. Luôn để ô
**nền xem** ở **màu áo**. Khoảng trống giữa các nét chữ vốn là nền, thấy checkerboard ở đó là đúng.

**Đặt vị trí mà để cỡ 100% thì không thấy gì đổi.** Ở cỡ 100% thiết kế đã lấp kín khung, neo vào góc
nào cũng như nhau vì không còn chỗ để dịch. Giảm cỡ xuống, ví dụ 26%. Trang có dòng nhắc màu vàng
ngay khi bạn rơi vào trường hợp này.

**Sửa code thì phải khởi động lại trang.** Trang được nạp vào bộ nhớ lúc chạy `./run.sh --ui`. Nhấn
`Ctrl+C` ở cửa sổ terminal rồi chạy lại. Nếu nó báo cổng đang bận, tức là vẫn còn một trang mở ở
`127.0.0.1:8765`.

### Khi có lỗi

Ảnh lỗi hiện thành một khối đỏ dưới cùng trang; bấm vào để mở đủ traceback, không phải một dòng cụt.
Ảnh vẫn nằm nguyên trong danh sách nên bạn đổi cờ rồi chạy lại ngay được.

Hai lỗi hay gặp nhất đều tự giải thích: chọn màu mực gần trùng màu nền ảnh thì bị từ chối vì in ra
không thấy gì, và bật `thân hình đặc` cho poster thì pipeline tự bỏ qua kèm dòng cảnh báo, file vẫn
ra đúng.

### Lấy file

Nút **Tải file in** tải bản đầy đủ 300 DPI, tên mang theo khung in và vị trí, xem mục
**Tên file in**. Khung 4500 x 5100 ở 300 DPI là 38,1 x 43,2 cm; đối chiếu với bàn in của xưởng trước
khi gửi.

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

Sau đó ảnh đi qua **raster** (mặc định): upscale 4 lần rồi co về khung in, giữ nguyên màu và chi
tiết. Cách upscale được chọn theo kiểu thiết kế, xem mục **Ba cách upscale**. Với **minh họa phẳng
thật sự** (mảng màu, không gradient), thêm `--vector` để gom màu rồi trace vector: mảng màu tuyệt đối
phẳng, viền cong mượt ở mọi kích cỡ.

## Ba cách upscale

| Kiểu | Khi nào | Cách |
|---|---|---|
| `flat` | chữ, logo, mảng màu đều | Real-ESRGAN model anime: mép sắc, không quầng |
| `detail` | ảnh chụp, tranh có gradient và texture mịn | Real-ESRGAN model x4plus: giữ chất liệu, hơi làm mượt |
| `grain` | halftone, chấm bi, vệt bắn, hạt sờn | **Lanczos thường**, không cho model bịa chi tiết |

Kiểu `grain` có vì một lý do cụ thể. Cả hai model Real-ESRGAN đều coi chấm bi halftone là nhiễu cần
"sửa": model x4plus biến chấm thành vệt lông xù, model anime biến thành mảng nứt vỡ. Lanczos thường
giữ chấm là chấm, chỉ mềm đi một chút. Nhận diện bằng cách đếm số mảnh mực rời nhỏ: trên 26 thiết
kế thật, bốn poster halftone và hai tấm vệt bắn có 3,4 đến 21 mảnh nhỏ trên 1000 pixel mực, còn ảnh
chụp và đồ họa phẳng dưới 0,9. Dòng log in `kiểu: grain`; ép tay bằng `--style grain` nếu nhận diện
sai.

Đổi lại, chấm bi qua Lanczos mềm hơn chấm bi bị model "sửa" thành khối: trên poster sọ halftone,
mực đặc giảm từ 50% xuống 26% và phủ thấp tăng từ 16% lên 21%. Con số xấu hơn nhưng ảnh đúng hơn,
vì phần "đặc" cũ là vệt lông xù model bịa ra. Với DTF, kết hợp `--dtf-safe` để chấm bi đủ to mà bám:
cùng tấm đó, nét mảnh từ 8,5% về 0, đốm nhỏ từ 2,5% về 0,9%, phủ thấp về lại 16%.

## Các tùy chọn

Hằng ngày chỉ cần ba cờ:

| Cờ | Mặc định | Khi nào dùng |
|---|---|---|
| `--shirt X` | same | `same` = áo cùng màu nền ảnh, `other` = áo khác màu, `auto` = nền đen coi là áo đen, còn lại áo khác màu. Xem bảng trên. |
| `--size WxH` | 4500x5100 | Kích thước file in. Số ≥ 200 là pixel, nhỏ hơn là cm (`30x40`). |
| `--preset X` | none | Vị trí in hay dùng, điền sẵn `--place` và `--scale`: `chest-left`, `chest-right`, `chest`, `back-neck`, `full`. Xem mục **Đặt hình nhỏ trên khung in**. |
| `--vector` | tắt | Minh họa phẳng: gom 12 màu rồi trace vector. Không dùng cho tranh có gradient, texture, chữ rất nhỏ. |
| `file1 file2 …` | | Chỉ xử lý các file này. |

Nâng cao, thường không cần:

| Cờ | Mặc định | Khi nào dùng |
|---|---|---|
| `--style X` | auto | Ép cách upscale: `flat` (model anime), `detail` (model x4plus), `grain` (Lanczos, cho halftone và vệt bắn). Xem mục **Ba cách upscale**. |
| `--dtf-safe` | tắt | Nới mọi nét và đốm mảnh hơn 0,5 mm ra đúng 0,5 mm bằng chính màu của nó, để in DTF không bong. Tên file thêm `_dtf-safe`. |
| `--icc FILE` | hồ sơ chung của macOS | Hồ sơ CMYK của xưởng in, dùng để đo màu ngoài gamut và xem như in. |
| `--colors N` | không gom | Gom về N màu (raster) hoặc đổi số màu khi `--vector` (mặc định 12). |
| `--merge D` | 12 | Ngưỡng gộp hai màu gần nhau khi gom (CIELAB ΔE). Tăng nếu còn đốm màu lệch, giảm nếu hai màu khác bị gộp. |
| `--dtf-warn P` | 5 | Cảnh báo khi quá P phần trăm diện tích mực nằm dưới 40% độ phủ, mức mà in DTF dễ bong. `100` = tắt. |
| `--no-clean` | tắt | Không dọn mực vô hình trước khi lưu. Mặc định có dọn. |
| `--fill-limit P` | 20 | Chốt chặn cho `--fill-holes`: nếu tô đặc làm tăng quá P phần trăm mực in đè lên áo cùng màu thì bỏ qua và cảnh báo. Đo trên chính bản sẽ in. `100` = tắt. |
| `--fill-floor D` | 0 | Sàn cho `--fill-holes`: chỗ ảnh gốc cách màu nền dưới D không tô đặc, để áo làm màu đó. Ảnh đen trắng trên nền đen thử `32`. Xem mục **Thân hình quá tối so với nền**. |
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
./run.sh --preset chest-left               # logo ngực trái người mặc, 26% khung
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

**Chốt chặn tự động.** Với poster, model cắt hình coi cả tấm là một khối và lấp luôn nền giữa các
chi tiết; mực đó in ra trùng màu áo, vừa phí vừa nổi thành mảng trên vải. Nên pipeline đo mức tăng
của loại mực đó và bỏ qua việc tô đặc nếu vượt `--fill-limit`, mặc định 20%, kèm một dòng cảnh báo.
Đo trên bộ ảnh thật: ảnh có người thật tăng 5 đến 11%, còn ba tấm poster tăng 27 đến 60%. Phép đo
làm trên **chính bản sẽ in**, sau upscale: ở ảnh gốc, nhiễu hạt đẩy vùng tối lên trên ngưỡng, còn
model upscale làm mịn nó về sát nền, và một tấm từng qua chốt ở 3% rồi ra file in 36%.

### Thân hình quá tối so với nền: `--fill-floor`

Ảnh chụp đen trắng trên nền đen là trường hợp khó: quần, bóng dưới cánh tay và nếp áo tối chỉ cách
nền vài mức, nhìn trên màn hình là đen. Tô đặc cả thân hình biến chúng thành một khối mực gần đen
in lên vải đen, tốn mực, dày bóng như nhựa, và mép silhouette lộ thành đường ánh trên vải. Không tô
thì mặt và cánh tay in mỏng.

`--fill-floor D` là điểm giữa: chỗ ảnh gốc cách màu nền dưới D không tô đặc mà để áo làm màu đó,
chỗ sáng hơn nền rõ vẫn tô. Mặc định 0 giữ hành vi cũ, tô cả thân hình. Với ảnh đen trắng thử `32`,
cùng mức với `--floor` của bước key. Xem bảng so sánh trên ảnh thật ở cuối mục này để chọn.

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

Các vị trí hay dùng có tên sẵn, `--preset` điền `--place` và `--scale` cho bạn. Tên theo **người
mặc**: ngực trái của người mặc nằm bên **phải** file in.

| Preset | Bằng | Dùng cho |
|---|---|---|
| `chest-left` | `--place top-right --scale 26` | logo ngực trái, khoảng 10 cm |
| `chest-right` | `--place top-left --scale 26` | logo ngực phải |
| `chest` | `--place top --scale 55` | ngực giữa, cỡ A4 |
| `back-neck` | `--place top --scale 20` | nhãn nhỏ sau gáy |
| `full` | `--place center --scale 100` | như mặc định |

`--place` hoặc `--scale` đặt tay vẫn thắng preset, ví dụ `--preset chest-left --scale 30` cho logo
ngực to hơn một chút. Tên file in mang vị trí và cỡ đã điền, không mang tên preset.

## In một màu: `--ink`

```bash
./run.sh --ink black          # tách một màu đen, cho áo sáng
./run.sh --ink "#ff0055"      # một màu hồng
```

Đây là bản tách màu của thợ in lụa, không phải bôi cả hình thành một khối. Mỗi pixel được hỏi nó đi
bao xa trên đường từ **màu áo** tới **màu mực**: chỗ trùng màu áo thì không có mực, chỗ tới hẳn màu
mực thì phủ kín, chỗ ở giữa ra độ phủ ở giữa. Nhờ vậy một cái sọ chụp ảnh cho ra đúng mảng đặc và khe
hở như khi làm tay trong Photoshop.

Câu hỏi đặt trên **ảnh đã ghép lên áo**, vì đó mới là thứ mắt thấy; màu lưu trong file đã được chia
ngược cho alpha nên tự nó không nói lên độ đậm nhạt.

Chọn mực gần trùng màu nền ảnh gốc thì pipeline báo lỗi thay vì lặng lẽ ra file rỗng, vì in ra sẽ
không thấy gì. Ảnh gốc đã trong suốt sẵn cũng bị từ chối: không có nền thì không đo được màu áo.

## Chấm cả lô: `--audit`

```bash
./run.sh --audit
```

In bảng ba con số cho mọi file in đang có rồi thoát, không xử lý ảnh nào. Trang UI cho từng ảnh một;
bảng này cho cả lô một lượt để thấy ảnh nào lệch khỏi phần còn lại. Ảnh bị đánh dấu `<--` kèm lý do
khi mực thừa quá 20%, sai số quá 5, hoặc mực đặc dưới 50%.

## Tên file in

```
tên-ảnh_khung-in_vị-trí.png       ->  skull_4500x5100_center.png
tên-ảnh_khung-in_vị-trí_cỡ.png    ->  skull_4500x5100_top-right_26pc.png
tên-ảnh_khung-in_vị-trí_mực.png   ->  skull_4500x5100_center_ink-black.png
…_cutout / _fill / _vector        ->  skull_4500x5100_center_fill.png
…_grain / _flat / _detail / _c8   ->  skull_4500x5100_center_grain.png
```

Mặc định thì tên gọn. Mọi cờ làm **đổi bức ảnh** đều để một đuôi trong tên, theo thứ tự cố định:
cỡ, màu mực, `cutout` (áo khác màu), `fill` (thân hình đặc), `vector`, kiểu upscale ép tay, `cN` gom
màu, `dtf-safe`. Nhờ vậy chạy cùng một ảnh có và không có `thân hình đặc` ra **hai file** nằm cạnh
nhau để so trên trang, thay vì lần sau đè lần trước. Đuôi `fill` chỉ xuất hiện khi thật sự đã tô
đặc; chốt chặn bỏ qua thì tên không có nó. Ảnh so sánh trong `review/` dùng đúng tên đó.

## File in nhớ cách nó được tạo

Mỗi file in ghi trong chính nó (một đoạn văn bản trong PNG, không ảnh hưởng gì đến việc in) bộ cờ
khác mặc định, nền đã nhận diện, cách xử lý đã chọn, và một dòng lệnh chạy lại ra đúng file này:

```
./run.sh --size 240x240 --fill-holes --preset chest-left
```

Trang hiện dòng đó dưới các con số là `cờ đã dùng`, và dùng nó để biết file là bản key, bản cắt
hình hay bản một màu mực rồi chấm cho đúng. File làm trước khi có mục này hiện `—` và được chấm như
bản key, vì đó là cách chúng đã được tạo. Đọc bằng Python: `pipeline.read_meta(path)`.

## Chuẩn bị file cho xưởng in

Mỗi file lưu ra đều được **dọn mực vô hình** và **gắn hồ sơ màu sRGB**.

Dọn bỏ hai thứ mắt không thấy nhưng máy in vẫn xử lý: pixel alpha dưới 8, và đốm vừa nhỏ hơn 0,5mm
vừa không chỗ nào đậm quá 40. Ngưỡng đậm chính là thứ giữ lại hạt halftone và vệt sờn cố ý, vì chúng
nhỏ nhưng đậm. Đo trên một poster in thật: bỏ 1,7 triệu pixel, tức 15,7% số pixel có mực, mà chỉ mất
**0,019%** tổng lượng mực và ảnh nhìn không khác. Tắt bằng `--no-clean`.

sRGB phải có: RIP gặp file không gắn hồ sơ sẽ tự đoán không gian màu, và màu in ra lệch so với thứ
bạn đã duyệt trên màn hình.

Dòng log cũng in khổ thật bằng cm, ví dụ `4500x5100 px @ 300 DPI = 38,1 x 43,2 cm`, để đối chiếu với
bàn in của xưởng trước khi gửi.

## Cảnh báo cho in DTF

Vùng mực dưới **40% độ phủ** nhận ít bột keo nên dễ bong sau vài lần giặt. In DTG có lót trắng thì
không sao, nhưng in DTF thì đáng lo.

Pipeline đo tỉ lệ đó cho từng file và in cảnh báo khi vượt 5%:

```
CẢNH BÁO in DTF: 15% diện tích mực nằm dưới 40% độ phủ (ngưỡng 5%). Vùng đó nhận ít bột keo
nên dễ bong; in thử một chiếc và giặt vài lần trước khi chạy số lượng.
```

Con số này cũng là một cột trong `./run.sh --audit` và hiện cạnh ba con số kia trên trang, tô vàng
khi vượt ngưỡng. Đổi ngưỡng bằng `--dtf-warn`, đặt `100` để tắt.

Đo trên một bộ 26 thiết kế: trung vị 0,9%, còn bốn tấm poster có halftone và quầng sáng rơi vào
9,2 đến 15,7%. Ngưỡng 5 nằm giữa hai nhóm đó.

## Cảnh báo ảnh gốc nhỏ

Model upscale làm nét được 4 lần. Ảnh 1024 px lên khung 4500 px là 4,4 lần, phần dư là kéo giãn
thường, viền mềm đi một chút. Pipeline in cảnh báo khi phải phóng quá 4 lần:

```
CẢNH BÁO ảnh gốc nhỏ: 1024x1024 px phải phóng 4,4 lần cho khung này, model chỉ làm nét được
4 lần, phần dư là kéo giãn. Nếu thấy mờ, sinh lại ảnh ở kích thước lớn hơn.
```

Với ảnh AI 1024 px in khổ mặc định thì mức 4,4 là thường gặp và vẫn in tốt; con số này đáng lo khi
lên 6 đến 8 lần, ví dụ ảnh 512 px hoặc ảnh đã cắt nhỏ.

## Nét mảnh và đốm nhỏ: `--dtf-safe`

DTF bám kém ở hai chỗ: nét mảnh hơn khoảng 0,5 mm và đốm rời nhỏ hơn khoảng 1 mm². Bột keo không
đủ diện tích để giữ, nên chúng bong hoặc rơi ngay trên bàn ép. Pipeline đo hai phần trăm đó trên
từng file in (cột `nét mảnh` và `đốm nhỏ`) và báo trong kết luận khi vượt 5% và 1%. Trên bộ 26
thiết kế, ba poster halftone có 8 đến 16% mực trong nét mảnh, phần còn lại dưới 3%.

`--dtf-safe` nới **chỉ** những nét và đốm mảnh hơn 0,5 mm ra đúng 0,5 mm, bằng chính màu của pixel
mực gần nhất. Mảng khối và mép của nó không đổi một pixel. Đây là "nét tối thiểu" thợ in lụa vẫn làm
tay: mất một chút chi tiết ở halftone, đổi lấy áo không bong sau khi giặt. Tên file thêm
`_dtf-safe` để bản gốc và bản nới không đè lên nhau; chạy `--audit` để so hai bản.

## Màu in được và màu không

Ảnh AI hay dùng hồng neon, đỏ tươi, xanh lá chói, là những màu **màn hình hiện được nhưng mực CMYK
không pha ra được**. In ra chúng xỉn đi rõ, và khách so với ảnh duyệt trên điện thoại sẽ thấy khác.

Pipeline đưa màu từng pixel mực qua một hồ sơ CMYK rồi về lại, đo lệch bao nhiêu. Cột `ngoài gamut`
là phần trăm mực lệch quá 25 ΔE, mức xỉn hẳn, kèm mức lệch lớn nhất. Trên trang, bật **xem như in**
để thấy toàn bộ file với màu đã qua hồ sơ, đặt cạnh ảnh gốc.

Ngưỡng 25 cố ý cao. Với hồ sơ chung của macOS, đỏ logo `(215, 8, 22)`, hồng neon `(255, 40, 130)`
và xanh Bills đều lệch 19 đến 20 ΔE, tức hồ sơ đó không phân biệt được "đỏ in được" với "hồng khó
in", vì nó hẹp hơn mực DTF thật. Nên chỉ những màu không mực nào pha ra mới bị đếm: xanh lá chói
lệch 75, xanh dương thuần 98, tím 60, đỏ 255 là 34. Mức lệch lớn nhất vẫn được báo để bạn biết màu
chủ đạo sẽ xỉn bao nhiêu, và **xem như in** cho thấy điều đó bằng mắt.

Hồ sơ đúng nhất là **ICC của chính xưởng in** cho bộ mực và loại film họ dùng; xin họ một file rồi
chạy `./run.sh --icc xuong.icc` (cả với `--ui`). Không có thì pipeline dùng hồ sơ CMYK chung của
macOS, là mức bi quan: mực DTF thật thường rộng hơn một chút. Máy không có hồ sơ nào thì cột này
hiện `—`.

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
4. **Xuất** PNG 300 DPI đặt đúng vị trí trên canvas theo `--place` và `--scale`, tên mang theo khung
   in và vị trí, kèm ảnh so sánh trong `review/` (ghép trên màu áo khi nền là đen, trắng hay màu
   trơn; còn lại xem trên nền ca-rô).

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
pipeline.py     toàn bộ logic xử lý và dòng lệnh
ui.py           trang xem tại máy: server HTTP, hàng chạy, đo chất lượng cho từng ảnh
ui.html         giao diện của trang, nạp vào bộ nhớ lúc chạy ./run.sh --ui
run.sh          ./run.sh [cờ] [file...]
setup.sh        cài một lần
examples/       ảnh mẫu để thử
tests/          pytest
docs/superpowers/   spec và kế hoạch triển khai
```

`output/` và `review/` không được dọn tự động và lớn nhanh (một file in 4500 x 5100 nặng 5 đến
30 MB). Xóa tay khi đã gửi xưởng; chạy lại từ `input/done/` bất cứ lúc nào cũng ra lại đúng file,
vì cờ đã ghi trong file in và trang nhớ cờ theo từng ảnh.
