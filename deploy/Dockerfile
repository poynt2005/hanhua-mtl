# syntax=docker/dockerfile:1
#
# hanhua GPU image: bakes the WHOLE stack so a rented box boots ready in ~1 min
# and never re-runs a fragile install (that was the #1 cause of wasted money).
#
#   torch (from base) + transformers(pinned) + manga-ocr + easyocr + LaMa
#   + comic-text-detector (from git; NOT on PyPI) + ollama + libGL
#   + OCR/inpaint weights pre-downloaded + a BUILD-TIME smoke test
#
# Build (CPU is fine; no GPU needed to build):
#   docker build -f deploy/Dockerfile -t <registry>/hanhua-gpu:1 deploy
# Optionally bake the 9 GB Sakura GGUF (adds ~9 GB, removes the boot-time pull):
#   docker build --build-arg BAKE_SAKURA=1 -f deploy/Dockerfile -t ... deploy
#
# Run on vast.ai:
#   --image <registry>/hanhua-gpu:1 -p 8000:8000 -p 8080:8080

ARG BASE=pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
FROM ${BASE}

ARG SAKURA_MODEL=hf.co/SakuraLLM/Sakura-GalTransl-14B-v3.8:Q4_K_M
ARG BAKE_SAKURA=0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/usr/local/bin:/opt/conda/bin:$PATH

# libgl1/libglib2.0-0 : opencv (cv2) needs libGL.so.1  -> EasyOCR import
# git                 : comic-text-detector lives on GitHub, not PyPI
# curl + zstd         : ollama install script
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 git curl ca-certificates zstd \
    && rm -rf /var/lib/apt/lists/*

# --- python deps -------------------------------------------------------------
# torch/torchvision come from the base image; a constraints file stops pip from
# silently replacing CUDA torch with a CPU wheel.
RUN printf 'torch==2.4.0\ntorchvision==0.19.0\n' > /tmp/constraints.txt
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
        "transformers==4.46.3" \
        "manga-ocr==0.1.16" \
        "easyocr==1.7.2" \
        "fastapi>=0.110" "uvicorn[standard]>=0.29" "python-multipart>=0.0.9" \
        "pillow>=10" "numpy<2" \
        simple-lama-inpainting httpx

# comic-text-detector is NOT on PyPI -> install from GitHub (best effort).
RUN pip install --no-cache-dir "git+https://github.com/dmMaze/comic-text-detector.git" \
    || echo "[warn] comic-text-detector unavailable; OCR server falls back to EasyOCR"

# --- ollama ------------------------------------------------------------------
RUN curl -fsSL https://ollama.com/install.sh | sh

# --- pre-download OCR / inpaint weights (so boots don't fetch them) -----------
RUN python - <<'PY'
from manga_ocr import MangaOcr
MangaOcr()
import easyocr
easyocr.Reader(["ja"], gpu=False)
from simple_lama_inpainting import SimpleLama
SimpleLama()
print("pre-downloaded manga-ocr / easyocr / LaMa weights")
PY

# --- optionally bake the Sakura GGUF (removes the boot-time 9 GB pull) --------
RUN if [ "${BAKE_SAKURA}" = "1" ]; then \
        (nohup ollama serve >/tmp/ollama.log 2>&1 &) && sleep 6 \
        && OLLAMA_HOST=127.0.0.1:11434 ollama pull "${SAKURA_MODEL}" \
        && pkill ollama || echo "[warn] sakura bake failed"; \
    else echo "skipping Sakura bake (BAKE_SAKURA=0)"; fi

COPY ocr_server.py /root/ocr_server.py
COPY container_start.sh /root/container_start.sh
RUN chmod +x /root/container_start.sh

# --- BUILD-TIME smoke test ---------------------------------------------------
# This is the whole point of the image: catch every dependency problem HERE,
# not on a paid GPU box. It must import the server and actually run /prepare.
RUN python - <<'PY'
import io, sys
sys.path.insert(0, "/root")
from fastapi.testclient import TestClient
from PIL import Image
import ocr_server

client = TestClient(ocr_server.app)
assert client.get("/health").status_code == 200

buf = io.BytesIO()
Image.new("RGB", (320, 480), (255, 255, 255)).save(buf, "PNG")
resp = client.post("/prepare", files={"file": ("t.png", buf.getvalue(), "image/png")})
assert resp.status_code == 200, resp.text
body = resp.json()
assert "blocks" in body and "cleaned" in body, body.keys()
print("BUILD SMOKE TEST OK: keys=%s" % list(body.keys()))
PY

EXPOSE 8000 8080
CMD ["/root/container_start.sh"]
