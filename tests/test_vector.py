import shutil

import pytest

import pipeline

pytestmark = pytest.mark.skipif(shutil.which("resvg") is None, reason="resvg not installed")


def test_trace_and_render(tmp_path, red_circle):
    png, svg = tmp_path / "c.png", tmp_path / "c.svg"
    red_circle.save(png)
    pipeline.trace_svg(png, svg)
    out = pipeline.render_svg(svg, (2000, 3000))
    assert out.size == (2000, 2000)
    assert out.getpixel((5, 5))[3] == 0
    assert out.getpixel((1000, 1000)) == (220, 30, 30, 255)
