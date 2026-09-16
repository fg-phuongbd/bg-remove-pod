# Sổ tay trang chạy

Bản đọc đẹp có ảnh chụp: https://claude.ai/artifact/7z2DKzYujQboccTzdXJsR8

```bash
cd ~/Phuong-data/tshirt-pipeline
./run.sh --ui
```

Trang tự mở ở `127.0.0.1:8765`, chạy tại máy, không gửi ảnh đi đâu.

## Năm bước

**1. Thêm ảnh.** Kéo thả vào bất cứ đâu trên trang, hoặc chép vào `input/` rồi tải lại trang.
Nhận png, jpg, webp. Ảnh gốc giữ nguyên tại chỗ nên chạy lại bao nhiêu lần cũng được.

**2. Chọn ảnh.** Thanh cờ trên cùng luôn hiện cờ *của riêng ảnh đang chọn*. Dưới tên mỗi thẻ
ghi những cờ khác mặc định. Trình duyệt nhớ cả khi đóng trang.

**3. Đặt cờ theo loại thiết kế.**

| Thiết kế | Đặt gì |
|---|---|
| Đồ họa phẳng: chữ, logo, huy hiệu | để nguyên mặc định |
| Poster vẽ tay, có glow hoặc halftone | để nguyên mặc định |
| Ảnh chụp người hoặc nhân vật | bật `thân hình đặc` |
| Ảnh chụp người kèm chữ đồ họa | bật `thân hình đặc` |
| Bản in nhỏ đặt ở một góc | `đặt` = top-right, `cỡ` = 26 |
| In một màu mực | `mực` = đen hoặc trắng |

Tám cờ hay dùng nằm trên thanh, sáu cờ còn lại trong khối `cờ nâng cao`.

**4. Chạy.** *Chạy ảnh đang chọn* làm một ảnh, *Chạy tất cả đang chờ* làm hết, mỗi ảnh dùng cờ
riêng. Ô `cùng lúc` đặt số ảnh song song, mặc định 2, đẩy lên 3 thì nhanh hơn rõ. Mỗi ảnh
khoảng 14 giây, hoặc 45 giây nếu bật `thân hình đặc`.

**5. Đọc kết luận.**

| Mức | Nghĩa |
|---|---|
| Đủ điều kiện in | không thấy vấn đề nào |
| In được, nên xem lại | phủ thấp vượt ngưỡng DTF, hoặc mực mỏng bất thường |
| Không dùng được | file rỗng, mực in đè lên áo cùng màu quá 20%, hoặc sai số khi in quá 5 |

## Bốn con số

| Số | Nghĩa | Đọc thế nào |
|---|---|---|
| mực đặc | phần trăm pixel đục hoàn toàn | So trong cùng loại thiết kế. Logo phẳng dưới 85% là đáng ngờ; poster halftone 42% vẫn đúng. |
| mực trùng màu áo | mực đục nhưng cùng màu áo | Dưới 5% bỏ qua. Trên 20% gần như chắc là bật `thân hình đặc` nhầm cho poster. |
| phủ thấp | mực dưới 40% độ phủ | Quan trọng nhất với **in DTF**: vùng đó ít bột keo nên dễ bong. Trên 5% thì in thử một chiếc, giặt vài lần rồi hãy chạy số lượng. |
| sai số khi in | ghép lên màu áo rồi so ảnh gốc, thang 0-255 | Dưới 1 mắt không thấy. Trên 5 thì mở `review/` xem bằng mắt. |

Nút **Chấm cả lô** hiện bảng này cho mọi file một lượt. Ngoài dòng lệnh: `./run.sh --audit`.

## Ba cái bẫy

**Đừng chấm trên nền trắng.** Thiết kế cho áo tối phần lớn là mực sáng, nên nét trắng biến mất
vào nền trắng và trông như thủng. Luôn để ô `nền xem` ở `màu áo`.

**Đặt vị trí mà để cỡ 100% thì không thấy gì đổi.** Thiết kế đã lấp kín khung, không còn chỗ để
dịch. Giảm cỡ xuống, ví dụ 26%.

**Sửa code thì phải khởi động lại trang.** Ctrl+C ở terminal rồi chạy lại. Báo cổng bận tức là
vẫn còn một trang đang mở.

## Khi có lỗi

Khối đỏ dưới cùng trang, bấm vào để mở đủ traceback. Ảnh vẫn nằm trong danh sách nên đổi cờ rồi
chạy lại được ngay.

## Lấy file

Nút **Tải file in** tải bản đầy đủ 300 DPI. Tên mang theo khung in và vị trí:

```
skull_4500x5100_center.png
skull_4500x5100_top-right_26pc.png
skull_4500x5100_center_ink-black.png
```

Khung 4500 x 5100 ở 300 DPI là 38,1 x 43,2 cm. Mỗi file đều đã dọn mực vô hình và gắn sRGB.
