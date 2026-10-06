#!/bin/bash
# Cài một lần. Chạy lại được nhiều lần, không hại gì.
set -e
cd "$(dirname "$0")"

echo "==> Homebrew: uv + resvg"
brew list uv >/dev/null 2>&1 || brew install uv
brew list resvg >/dev/null 2>&1 || brew install resvg

echo "==> Python env (uv sync)"
uv sync

echo "==> Thư mục làm việc"
mkdir -p input/done input/failed output review work bin

echo "==> Real-ESRGAN (chỉ cần cho --raster)"
if [ ! -x bin/realesrgan-ncnn-vulkan ]; then
  URL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-macos.zip"
  if curl -fL "$URL" -o bin/re.zip && unzip -oq bin/re.zip -d bin && chmod +x bin/realesrgan-ncnn-vulkan; then
    rm -f bin/re.zip
    xattr -dr com.apple.quarantine bin 2>/dev/null || true
    echo "    OK"
  else
    echo "    CẢNH BÁO: không tải được Real-ESRGAN. --raster sẽ dùng Lanczos thay thế."
  fi
else
  echo "    đã có"
fi

echo "==> Model ảnh chụp cho thân người (4xLSDIRplusC, khi bật thân hình đặc)"
if [ ! -f bin/models/4xLSDIRplusC.bin ]; then
  BASE="https://raw.githubusercontent.com/upscayl/custom-models/main/models"
  mkdir -p bin/models
  if curl -fL "$BASE/4xLSDIRplusC.param" -o bin/models/4xLSDIRplusC.param && \
     curl -fL "$BASE/4xLSDIRplusC.bin" -o bin/models/4xLSDIRplusC.bin; then
    echo "    OK"
  else
    rm -f bin/models/4xLSDIRplusC.param bin/models/4xLSDIRplusC.bin
    echo "    CẢNH BÁO: không tải được. Thân người sẽ dùng x4plus như phần còn lại."
  fi
else
  echo "    đã có"
fi

echo "==> Kiểm tra"
./run.sh --help
echo
echo "Xong. Thả ảnh vào input/ rồi chạy ./run.sh"
