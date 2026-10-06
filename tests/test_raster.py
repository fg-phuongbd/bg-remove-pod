from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import pipeline


def test_upscale_fallback(monkeypatch):
    monkeypatch.setattr(pipeline, "REALESRGAN_BIN", Path("/nonexistent/realesrgan"))
    out = pipeline.upscale(Image.new("RGBA", (100, 50), (255, 0, 0, 255)))
    assert out.size == (400, 200)


def test_flatten_raster(red_circle):
    px = red_circle.load()
    for x in range(60, 340, 3):
        px[x, 200] = (230, 40, 40, 255)
    out = pipeline.flatten_raster(red_circle, colors=2)
    assert len({p[:3] for p in out.get_flattened_data() if p[3] == 255}) <= 2


def test_tighten_alpha():
    im = Image.new("RGBA", (3, 1))
    im.putpixel((0, 0), (0, 0, 0, 90)); im.putpixel((1, 0), (0, 0, 0, 128)); im.putpixel((2, 0), (0, 0, 0, 170))
    out = pipeline.tighten_alpha(im)
    assert out.getpixel((0, 0))[3] == 0 and out.getpixel((2, 0))[3] == 255
    assert 100 < out.getpixel((1, 0))[3] < 160


def _halftone(size=240, spacing=6, r=2):
    """Chấm bi trắng đều trên nền trong suốt: thứ Real-ESRGAN biến thành vệt lông xù."""
    from PIL import Image, ImageDraw
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for y in range(spacing, size - spacing, spacing):
        for x in range(spacing, size - spacing, spacing):
            d.ellipse((x - r, y - r, x + r, y + r), fill=(240, 240, 240, 255))
    return im


def test_is_grainy_tells_halftone_from_flat_and_photo(red_circle):
    import numpy as np
    from PIL import Image
    import pipeline

    assert pipeline.is_grainy(_halftone()) is True
    assert pipeline.is_grainy(red_circle) is False
    rng = np.random.default_rng(0)                                   # ảnh chụp: gradient + nhiễu nhẹ
    a = np.dstack([np.tile(np.linspace(40, 220, 240), (240, 1))] * 3 + [np.full((240, 240), 255)])
    a[:, :, :3] += rng.normal(0, 3, (240, 240, 3))
    assert pipeline.is_grainy(Image.fromarray(a.clip(0, 255).astype(np.uint8), "RGBA")) is False


def test_detect_style_names_grain_before_flat_or_detail():
    import pipeline

    assert pipeline.detect_style(_halftone()) == "grain"


def test_lanczos_upscale_never_calls_the_binary(monkeypatch, red_circle):
    import subprocess
    import pipeline

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("gọi binary")))
    out = pipeline.upscale(red_circle, model="lanczos")
    assert out.size == (1600, 1600)


def test_is_grainy_ignores_noise_around_half_alpha_in_a_dark_photo():
    """Ảnh chụp người tối trên nền đen: key độ sáng cho da và tóc alpha quanh 120 ± 20. Đếm mảnh
    ở đúng ngưỡng 128 thì vùng đó vỡ thành hàng nghìn đốm và ảnh chụp bị coi là hạt (Romo đo 3,1
    trên 1000, ngang poster halftone). Đếm ở ngưỡng thấp hơn thì thân người liền lại."""
    import numpy as np
    from PIL import Image
    import pipeline

    rng = np.random.default_rng(3)
    a = np.zeros((300, 300, 4), np.uint8)
    a[40:260, 60:240, :3] = 200
    a[40:260, 60:240, 3] = rng.normal(120, 20, (220, 180)).clip(0, 255)     # da tối, alpha quanh 128
    assert pipeline.is_grainy(Image.fromarray(a, "RGBA")) is False
    assert pipeline.is_grainy(_halftone()) is True                          # halftone thật vẫn là hạt


def _blurred_dot(peak, color=(200, 50, 50), rim_color=None, size=80, radius=12, blur=2.0):
    """Một chấm mực sau khi Lanczos phóng khoảng 3,6 lần: ruột đậm, mép là dốc mờ dài vài pixel."""
    import numpy as np
    from PIL import ImageDraw, ImageFilter
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).ellipse((size / 2 - radius, size / 2 - radius, size / 2 + radius, size / 2 + radius), fill=255)
    a = np.asarray(m.filter(ImageFilter.GaussianBlur(blur))).astype(np.float32) * peak / 255
    rgba = np.zeros((size, size, 4), np.uint8)
    rgba[..., :3] = color
    if rim_color is not None:
        rgba[a < peak * 0.9, :3] = rim_color      # màu đã chia ngược alpha ở mép: màu thuần, sáng hơn ruột
    rgba[..., 3] = a.round().astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def test_harden_dots_gives_a_blurred_dot_a_crisp_edge_at_half_its_peak():
    import numpy as np
    dot = _blurred_dot(255)
    out = np.asarray(pipeline.harden_dots(dot, 3.6))[..., 3].astype(int)
    a = np.asarray(dot)[..., 3].astype(int)
    soft = lambda x: int(((x > 8) & (x < 247)).sum())
    assert soft(out) < soft(a) / 3                      # dải mờ mỏng đi hẳn
    assert out[40, 40] == a[40, 40]                     # ruột giữ nguyên
    assert out[a < 0.35 * 255].max() == 0               # chân mờ dưới một nửa bị cắt
    area = int((out > 127).sum()); contour = int((a > 127).sum())
    assert abs(area - contour) <= 0.1 * contour         # mép nằm ở đường một nửa độ đậm


def test_harden_dots_keeps_a_dim_dot_dim():
    """Chấm mờ vẫn là chấm mờ: đỉnh 100 không bị xóa, cũng không bị nâng lên đặc."""
    import numpy as np
    out = np.asarray(pipeline.harden_dots(_blurred_dot(100), 3.6))[..., 3]
    assert 95 <= out.max() <= 100
    assert (out > 50).sum() > 300


def test_harden_dots_keeps_every_pixels_own_color():
    """Chép màu từ pixel ruột gần nhất biến vùng phun sơn thưa thành các ô màu phẳng."""
    import numpy as np
    dot = _blurred_dot(255, rim_color=(255, 255, 255))
    out = np.asarray(pipeline.harden_dots(dot, 3.6))
    assert (out[..., :3] == np.asarray(dot)[..., :3]).all()


def _lanczos_disk(alpha=215, r_src=14, n=40, grow=3.6):
    """Một mảng mực mép sắc ở ảnh gốc, phóng bằng Lanczos: sát mép có dải vọt cao hơn ruột."""
    from PIL import ImageDraw
    s = Image.new("RGBA", (n, n), (200, 50, 50, 0)); c = n / 2
    ImageDraw.Draw(s).ellipse((c - r_src, c - r_src, c + r_src, c + r_src), fill=(200, 50, 50, alpha))
    return s.resize((round(n * grow),) * 2, Image.Resampling.LANCZOS)


def test_harden_dots_does_not_ring_a_shape_with_a_bright_edge():
    """Lanczos vọt lên tới 240 sát mép một mảng 215. Lấy chỗ vọt làm đỉnh rồi nâng cả dải mép lên
    đó thì mép sáng hơn ruột thành một đường viền (từng ra 252). Mép chỉ được lên tới mức ruột."""
    import numpy as np
    big = _lanczos_disk()
    a = np.asarray(big)[..., 3].astype(int)
    out = np.asarray(pipeline.harden_dots(big, 3.6))[..., 3].astype(int)
    ramp = (a > 20) & (a < 200)                          # dải dốc mờ của mép
    raised = ramp & (out > a)
    assert raised.sum() > 50                             # mép có được làm cứng
    assert out[raised].mean() <= 215 + 10


def test_harden_dots_leaves_a_wide_glow_alone():
    """Glow trải dài hàng trăm pixel không phải mép mờ do phóng to."""
    import numpy as np
    ramp = np.tile(np.linspace(255, 0, 400), (40, 1))
    rgba = np.zeros((40, 400, 4), np.uint8); rgba[..., :3] = 240; rgba[..., 3] = ramp.round()
    out = np.asarray(pipeline.harden_dots(Image.fromarray(rgba, "RGBA"), 3.6))[..., 3].astype(int)
    body = ramp > 40                                    # bỏ phần đuôi rất mờ
    assert np.abs(out - rgba[..., 3].astype(int))[body].mean() < 2


def _fade_beside_a_block():
    """Khối đặc trắng, viền khử răng cưa 1 px, rồi một vùng glow mờ dần từ 39% xuống 0 rộng 200 px."""
    a = np.zeros((200, 400), np.float32)
    a[:, :100] = 255
    a[:, 100] = 128                                         # mép khử răng cưa của khối
    a[:, 101:301] = np.linspace(101, 0, 200)[None, :]       # glow dưới 40% độ phủ
    rgba = np.dstack([np.full((200, 400, 3), 240, np.uint8), a.round().astype(np.uint8)])
    return Image.fromarray(rgba, "RGBA")


def test_halftone_fade_turns_a_faint_glow_into_solid_dots():
    img = _fade_beside_a_block()
    out = np.asarray(pipeline.halftone_fade(img))
    a_in, a = np.asarray(img)[:, :, 3].astype(float), out[:, :, 3].astype(float)
    # Cách khối hơn 15 px: sát khối, cửa sổ đo độ mượt trùm cả mép khối nên glow ở đó giữ nguyên mượt.
    band = (slice(None), slice(116, 301))
    vals = np.unique(a[band])
    assert set(vals) <= {0.0, 255.0}, vals                  # chỉ còn chấm đặc hoặc vải
    keep = a_in[band] >= 0.15 * 255
    ratio = a[band][keep].mean() / a_in[band][keep].mean()
    assert 0.85 < ratio < 1.15, ratio                       # nhìn từ xa vẫn cùng độ đậm
    assert a[band][~keep].max() == 0                        # chấm quá nhỏ để bám keo thì bỏ
    assert (out[:, :101] == np.asarray(img)[:, :101]).all() # khối đặc và mép của nó không đổi
    assert (out[:, :, :3] == 240).all()                     # màu không đổi, chỉ alpha


def test_halftone_fade_leaves_a_solid_design_alone():
    im = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse((10, 10, 110, 110), fill=(200, 30, 40, 255))
    im = im.resize((360, 360), Image.Resampling.LANCZOS)     # mép mềm như sau khi phóng
    assert (np.asarray(pipeline.halftone_fade(im)) == np.asarray(im)).all()


def test_halftone_fade_leaves_a_noisy_texture_alone():
    """Vân hạt mờ (ảnh đen trắng in trên nền đen) không phải vùng mờ mượt: chấm hóa nó chỉ rắc chấm
    trắng ngẫu nhiên như tuyết lên mảng tối. Chỉ chỗ mượt mới thành chấm."""
    rng = np.random.default_rng(7)
    a = np.clip(rng.normal(45, 30, (200, 200)), 0, 101).astype(np.uint8)
    img = Image.fromarray(np.dstack([np.full((200, 200, 3), 230, np.uint8), a]), "RGBA")
    out = np.asarray(pipeline.halftone_fade(img))[:, :, 3]
    changed = (out != a).mean()
    assert changed < 0.05, changed
