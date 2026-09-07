# Tinh chỉnh viền cho đường tách nền AI

## Vấn đề

BiRefNet tìm "vật thể nổi bật". Với thiết kế có rìa vẽ thẳng lên nền màu trơn (tia sét, vết bắn
mực, quầng glow), model không thấy ranh giới: mask ôm theo khối, giữ một dải nền đục quanh hình và
cắt cụt chi tiết mảnh. Key màu (`--bg color`) sửa được viền nhưng làm trong suốt luôn các vùng
vẽ bằng màu nền ở sâu bên trong, nên chỉ đúng khi in áo cùng màu nền.

## Mục tiêu

Ảnh nền màu trơn, in lên áo **khác** màu: viền sạch như key màu, bên trong đặc như BiRefNet.
Tương đương Refine Edge + Decontaminate Colors trong Photoshop, nhưng tự động.

## Cách làm

`refine_edge(no_bg, original, band=0.05)`, chạy ngay sau `remove_bg` khi cắt hình trên nền màu trơn
(`resolve_bg` trả về refine = True cho nền `color` với áo khác màu):

1. `mask` = alpha của BiRefNet ≥ 128.
2. `r` = 5 % cạnh ngắn (≈ 63 px ở 1254 px). `core` = mask co vào r. `halo` = mask nở ra r.
3. `key` = `key_color(original)`: alpha theo khoảng cách tới màu nền lấy từ vành ảnh, màu đã bù.
4. Kết quả: trong `core` alpha 255 và giữ màu gốc (BiRefNet được tin ở sâu bên trong);
   trong dải `halo − core` lấy alpha và màu của `key` (từng pixel tự quyết định theo màu);
   ngoài `halo` alpha 0 (không nhặt nhiễu xa hình).
5. `--fill-holes` nếu có chạy sau bước này.

Co/nở dùng phép erode nhị phân sẵn có, đệm viền để không cuốn qua mép ảnh.

## Không làm

- Tự bật cho cả nền trắng và nền màu trơn khi cắt hình, không có cờ riêng. Với nền trắng, rủi ro
  "chi tiết trắng gần viền bị key mất" được xử lý bằng cách lấp các lỗ nhỏ (≤ (2r)² px) trong mask
  model trước khi co lõi: lỗ nhỏ là model nhầm chi tiết trắng thành nền, lỗ lớn là vùng trắng thật
  bên trong nét viền (line art). Quyết định sau khi 4 ảnh line art nền trắng bị viền trắng lem.
- Không đổi đường `black`, `white`, `color`, `none`.
- Không thêm tham số độ rộng dải ra CLI cho tới khi cần.

## Kiểm chứng

- Test đơn vị trên ảnh tổng hợp: vùng màu nền ở sâu bên trong giữ đục; nền màu nằm trong mask
  nhưng ở dải viền thành trong suốt; chi tiết mảnh ngoài mask được lấy lại; nền xa hình vẫn trong.
- Đo trên ảnh hồng thật, so với BiRefNet thuần và key màu: pixel nền còn đục, thiết kế mất,
  độ trung thực khi ghép lên áo hồng và áo đen. Kiểm tra hồi quy trên ảnh mascot nền trắng.

## Cập nhật 2026-09-07: gom cờ về `--shirt`

Cờ `--refine` bị bỏ cùng lúc `--bg` thành cờ ẩn. Người dùng chỉ trả lời `--shirt same|other`;
`detect_bg` phân loại nền thành `none|black|white|color`, `resolve_bg(kind, shirt)` cho ra cách xử lý
và có tinh chỉnh viền hay không. Mặc định `auto`: nền đen coi là áo đen, còn lại coi là áo khác màu,
giữ đúng hành vi lô cũ.

## Cập nhật 2026-09-07 (chiều): nền trắng

4 ảnh line art hồng trên nền trắng cho thấy BiRefNet để lại dải trắng mờ bám theo nét và mảnh trắng
kẹt giữa các nét, in lên áo tối thành vệt trắng. Tinh chỉnh viền bật cho nền trắng, kèm lấp lỗ nhỏ
trong mask để giữ mắt, răng trắng của minh họa (kiểm chứng trên mascot). Cùng lúc, `detect_style`
coi thiết kế có ≤ 8 màu thô là phẳng, vì line art gần như toàn pixel viền nên phép đo "thân phẳng" cũ
không có gì để đo và chọn nhầm model upscale x4plus làm nét bị uốn.
