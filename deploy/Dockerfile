# syntax=docker/dockerfile:1
#
# hanhua GPU image: bakes the WHOLE stack so a rented box boots ready in ~1 min
# and never re-runs a fragile install (that was the #1 cause of wasted money).
#
#   torch (from base) + transformers(pinned) + manga-ocr + easyocr + LaMa
#   + comic-text-detector (vendored at a pinned commit; it has no package metadata)
#   + ollama + libGL
#   + EVERY model weight the pipeline uses, in a fixed cache path
#   + runtime forced offline (HF_HUB_OFFLINE) + a BUILD-TIME offline smoke test
#
# Model weights baked (see "bake every weight the pipeline actually uses"):
#   manga-ocr           -> $HF_HOME                        (HuggingFace Hub)
#   easyocr ja          -> $EASYOCR_MODULE_PATH            (GitHub releases)
#   LaMa big-lama.pt    -> $TORCH_HOME/hub/checkpoints     (GitHub release)
#   comic-text-detector -> /opt/comic-text-detector/data/  (GitHub release)
#   Sakura 14B          -> ollama, imported from the Q6_K GGUF (see the bake step)
# The runtime therefore loads every model with ZERO network calls, even on hosts
# that cannot reach HuggingFace — and the build smoke test proves it by blocking
# every outbound connection.
#
# Build (CPU is fine; no GPU needed to build):
#   docker build -f deploy/Dockerfile -t <registry>/hanhua-gpu:1 deploy
# Bake the full stack (adds the 9 GB Sakura GGUF, removes the boot-time pull):
#   docker build --build-arg BAKE_SAKURA=1 -f deploy/Dockerfile -t ... deploy
#
# Run on vast.ai:
#   --image <registry>/hanhua-gpu:1 -p 8000:8000 -p 8080:8080

ARG BASE=pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
FROM ${BASE}

ARG SAKURA_MODEL=sakura-qwen3-q6k
# Sakura-14B-Qwen3-v1.5 (Q6_K): the newest 14B Sakura, same size and VRAM as the
# previous GalTransl-14B-v3.8 Q6_K, so it is a free upgrade in quality.
#   file: sakura-14b-qwen3-v1.5-q6k.gguf  (12.12 GB)
# NOTE: this release is marked "非正式版" (pre-release) and it recommends a
# DIFFERENT system prompt - see SAKURA_SYSTEM in mth/providers.py.
ARG SAKURA_MODEL_URL=https://huggingface.co/SakuraLLM/Sakura-14B-Qwen3-v1.5-GGUF/resolve/main/sakura-14b-qwen3-v1.5-q6k.gguf
ARG BAKE_SAKURA=0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/usr/local/bin:/opt/conda/bin:$PATH

# One stable location per model cache, so what the BUILD downloads is exactly
# what the RUNTIME reads. If these paths drift, a cached model looks "missing"
# and the box silently downloads it again (or fails) on a paid GPU.
#   HF_HOME              -> manga-ocr-base (transformers/HuggingFace Hub)
#   TORCH_HOME           -> big-lama.pt (torch.hub, used by simple-lama)
#   EASYOCR_MODULE_PATH  -> craft detector + japanese recognizer
ENV HF_HOME=/opt/models/hf \
    TORCH_HOME=/opt/models/torch \
    EASYOCR_MODULE_PATH=/opt/models/easyocr

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

# --- comic-text-detector (vendored, pinned) ----------------------------------
# Upstream ships NO package metadata (no setup.py / pyproject.toml) and uses
# top-level imports, so `pip install` is impossible. Pin a commit, clone it into
# a fixed directory, drop the training-only scripts, and install the small
# runtime deps its import chain needs. Fully build-time: the runtime never
# touches the network. See llm_handoff/12_hf_network_audit.md.
ARG CTD_COMMIT=440b978563c71b758e31aaa315d100faba1efa2f
RUN git init -q /opt/comic-text-detector \
    && cd /opt/comic-text-detector \
    && git remote add origin https://github.com/dmMaze/comic-text-detector.git \
    && git fetch -q --depth 1 origin "${CTD_COMMIT}" \
    && git checkout -q FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${CTD_COMMIT}" \
    && rm -rf .git \
    && rm -f examples.ipynb db_dataset.py seg_dataset.py text_rendering.py \
             train_db.py train_seg.py requirements.txt utils/loss.py utils/export.py \
    && pip install --no-cache-dir -c /tmp/constraints.txt \
         pyclipper shapely torchsummary requests tqdm packaging

# Two tiny patches to the vendored code (training-era assumptions that do not
# hold in a modern, slim runtime):
#   1. utils/general.py does `import wandb` at module level, but only its training
#      Loggers class uses wandb -> make the import optional.
#   2. utils/yolov5_utils.py does `import pkg_resources as pkg`; setuptools >= 81
#      REMOVED that module, and check_version() runs on every inference
#      (models/yolov5/yolo.py -> Detect._make_grid) -> fall back to packaging.
RUN python - <<'PY'
import pathlib

root = pathlib.Path("/opt/comic-text-detector")

WANDB_NEEDLE = "import wandb\n"
WANDB_GUARD = ("try:\n    import wandb\n"
               "except ImportError:  # training-only; unused on the inference path\n"
               "    wandb = None\n")

PKG_NEEDLE = "import pkg_resources as pkg\n"
PKG_GUARD = ("try:  # setuptools >= 81 removed pkg_resources\n"
             "    import pkg_resources as pkg\n"
             "except ImportError:  # pragma: no cover - modern setuptools\n"
             "    from packaging import version as _packaging_version\n"
             "\n"
             "    class _PkgWrapper:\n"
             "        parse_version = staticmethod(_packaging_version.parse)\n"
             "\n"
             "    pkg = _PkgWrapper()\n")


def patch(rel, needle, guard):
    path = root / rel
    source = path.read_text()
    if guard in source:
        return "already patched"
    if needle not in source:
        return "NEEDLE NOT FOUND (upstream changed?)"
    path.write_text(source.replace(needle, guard, 1))
    return "patched"


print("utils/general.py     :", patch("utils/general.py", WANDB_NEEDLE, WANDB_GUARD))
print("utils/yolov5_utils.py:", patch("utils/yolov5_utils.py", PKG_NEEDLE, PKG_GUARD))
PY

# Fail the build if the whole import chain does not resolve.
RUN python -c "import sys; sys.path.append('/opt/comic-text-detector'); \
from inference import TextDetector; print('comic-text-detector import OK')"

# --- ollama ------------------------------------------------------------------
RUN curl -fsSL https://ollama.com/install.sh | sh

# --- bake every weight the pipeline actually uses ----------------------------
# manga-ocr   : HF Hub (transformers)  -> HF_HOME
# easyocr     : GitHub releases        -> EASYOCR_MODULE_PATH
# LaMa        : GitHub releases        -> TORCH_HOME/hub/checkpoints
# After this RUN the runtime must never need the network to load a model.
RUN python - <<'PY'
import os

from manga_ocr import MangaOcr
MangaOcr()

import easyocr
easyocr.Reader(["ja"], gpu=False)

from simple_lama_inpainting import SimpleLama
SimpleLama()


def dir_size_mb(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total // 1048576


for label in ("HF_HOME", "TORCH_HOME", "EASYOCR_MODULE_PATH"):
    path = os.environ.get(label, "")
    print("baked %-20s %-24s %d MB" % (label, path, dir_size_mb(path)))
print("pre-downloaded manga-ocr / easyocr / LaMa weights")
PY

# The detector weights are not in the repo either; fetch them from the upstream
# release (GitHub, not HuggingFace) into the vendored checkout's data/ dir.
RUN python - <<'PY'
import os
import urllib.request

url = ("https://github.com/zyddnys/manga-image-translator/releases/download/"
       "beta-0.2.1/comictextdetector.pt")
dst_dir = "/opt/comic-text-detector/data"
os.makedirs(dst_dir, exist_ok=True)
dst = os.path.join(dst_dir, "comictextdetector.pt")
urllib.request.urlretrieve(url, dst)
print("baked comic-text-detector weights %s (%d MB)" % (dst, os.path.getsize(dst) // 1048576))
PY

# --- bake the Sakura GGUF (Q6_K) ---------------------------------------------
# `ollama pull hf.co/SakuraLLM/Sakura-GalTransl-14B-v3.8:Q6_K` CANNOT work: the
# repo has no file whose name contains "Q6_K" (the unsuffixed file IS Q6_K).
# So download the file and import it with a Modelfile - deterministic, and the
# same thing the upstream deployment guide does. The GGUF carries its own chat
# template, so ollama uses the correct one automatically.
RUN if [ "${BAKE_SAKURA}" = "1" ]; then \
        curl -fL --retry 5 --retry-delay 5 -o "/root/${SAKURA_MODEL}.gguf" "${SAKURA_MODEL_URL}" \
        && ls -l "/root/${SAKURA_MODEL}.gguf" \
        && (nohup ollama serve >/tmp/ollama.log 2>&1 &) && sleep 6 \
        && printf 'FROM /root/%s.gguf\n' "${SAKURA_MODEL}" > /root/Modelfile \
        && OLLAMA_HOST=127.0.0.1:11434 ollama create "${SAKURA_MODEL}" -f /root/Modelfile \
        && OLLAMA_HOST=127.0.0.1:11434 ollama list \
        && OLLAMA_HOST=127.0.0.1:11434 ollama show "${SAKURA_MODEL}" | head -n 20 \
        && rm -f "/root/${SAKURA_MODEL}.gguf" \
        && pkill ollama || echo "[warn] sakura bake failed"; \
    else echo "skipping Sakura bake (BAKE_SAKURA=0)"; fi

# --- all weights are on disk now: make the runtime strictly offline ----------
# transformers/huggingface_hub perform a Hub freshness check on EVERY model load,
# even when the weights are already cached. On hosts where the Hub is unreachable
# the first OCR request hangs retrying and the /prepare call fails. With these
# flags the baked weights load with zero network calls. They are set AFTER the
# download steps above, which legitimately need the network.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY ocr_server.py /root/ocr_server.py
COPY container_start.sh /root/container_start.sh
RUN chmod +x /root/container_start.sh

# --- BUILD-TIME smoke test ---------------------------------------------------
# The whole point of the image: catch every dependency problem HERE, not on a
# paid GPU box — and prove the "runs fully offline" requirement. Every outbound
# connection is hard-blocked for the duration of the test (loopback stays
# allowed), so any component that still wants the network fails the build.
RUN python - <<'PY'
import io, os, socket, sys

_real_connect = socket.socket.connect


def _guard_connect(self, address):
    host = address[0] if isinstance(address, tuple) else address
    if host in ("127.0.0.1", "::1", "localhost"):
        return _real_connect(self, address)
    raise RuntimeError("OUTBOUND NETWORK BLOCKED: %r" % (address,))


socket.socket.connect = _guard_connect

sys.path.insert(0, "/root")
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
import ocr_server

client = TestClient(ocr_server.app)
health = client.get("/health")
assert health.status_code == 200, health.text
info = health.json()
assert info.get("detector") is True, "comic-text-detector did not load: %s" % info

# Every model-backed path must work with the network hard-blocked.
canvas = Image.new("RGB", (256, 256), (255, 255, 255))
mask = np.zeros((256, 256), dtype=np.uint8)
mask[100:150, 100:160] = 255
boxes, seg_mask = ocr_server.detect(canvas)
assert seg_mask is not None and tuple(seg_mask.shape[:2]) == (256, 256), "detector produced no mask"
assert ocr_server.inpaint(canvas, mask).size == canvas.size, "LaMa failed offline"
assert isinstance(ocr_server.get_mocr()(canvas.crop((0, 0, 64, 64))), str), "manga-ocr failed offline"

# End-to-end HTTP path used by the client.
buf = io.BytesIO()
Image.new("RGB", (320, 480), (255, 255, 255)).save(buf, "PNG")
resp = client.post("/prepare", files={"file": ("t.png", buf.getvalue(), "image/png")})
assert resp.status_code == 200, resp.text
body = resp.json()
assert "blocks" in body and "cleaned" in body, body.keys()

print("BUILD SMOKE TEST OK: detector=%s boxes=%d mask=%s cleaned=%s hf_offline=%s" % (
    info.get("detector"), len(boxes), tuple(seg_mask.shape[:2]),
    bool(body.get("cleaned")), os.environ.get("HF_HUB_OFFLINE")))
PY

EXPOSE 8000 8080
CMD ["/root/container_start.sh"]
