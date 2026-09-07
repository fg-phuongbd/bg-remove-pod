import shutil
from pathlib import Path

import pytest
from PIL import Image

import pipeline


def test_parse_args_defaults():
    a = pipeline.parse_args([])
    assert a.vector is False and a.colors is None and a.size == "4500x5100" and a.bg == "auto" and a.style == "auto"
    assert a.keep_input is False and a.files == []


def test_parse_args_flags():
    a = pipeline.parse_args(["--vector", "--colors", "6", "--size", "21x29.7", "--keep-input", "a.png"])
    assert a.vector and a.colors == 6 and a.size == "21x29.7" and a.keep_input and a.files == ["a.png"]


@pytest.mark.skipif(shutil.which("resvg") is None, reason="resvg not installed")
def test_main_batch_isolates_failures(tmp_path, monkeypatch, red_circle):
    for name in ("INPUT_DIR", "OUTPUT_DIR", "REVIEW_DIR", "WORK_DIR"):
        d = tmp_path / name.lower()
        d.mkdir()
        monkeypatch.setattr(pipeline, name, d)
    monkeypatch.setattr(pipeline, "remove_bg", lambda img: img)
    monkeypatch.setattr(pipeline, "check_tools", lambda: None)
    red_circle.save(pipeline.INPUT_DIR / "ok.png")
    (pipeline.INPUT_DIR / "bad.png").write_text("not an image")

    rc = pipeline.main(["--vector", "--size", "1181x1000"])

    assert rc == 1
    out = Image.open(pipeline.OUTPUT_DIR / "ok.png")
    assert out.size == (1181, 1000)
    assert round(out.info["dpi"][0]) == 300
    assert (pipeline.INPUT_DIR / "done" / "ok.png").exists()
    assert (pipeline.INPUT_DIR / "failed" / "bad.png").exists()
    assert (pipeline.REVIEW_DIR / "ok.png").exists()
    assert not (pipeline.INPUT_DIR / "ok.png").exists()


def test_parse_args_fill_holes():
    assert pipeline.parse_args(["--fill-holes"]).fill_holes is True
    assert pipeline.parse_args([]).fill_holes is False


def test_parse_args_bg_color():
    assert pipeline.parse_args(["--bg", "color"]).bg == "color"
