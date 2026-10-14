# hanhua-gpu

Prebuilt GPU image for the **hanhua** comic/manga translation pipeline.

It packages the whole GPU-side stack so a rented machine boots **ready in about
a minute** — no `apt`/`pip` install on the paid box, no dependency surprises.

Source code and documentation: https://github.com/poynt2005/hanhua-mtl

## What is inside

| Component | Purpose |
|---|---|
| CUDA + PyTorch runtime base | GPU compute |
| `manga-ocr` | Japanese manga OCR |
| `easyocr` | text-region detection (fallback detector) |
| `comic-text-detector` | segmentation-capable text detection (from its code host) |
| `simple-lama-inpainting` (LaMa) | neural text removal |
| `ollama` | OpenAI-compatible LLM server |
| `libgl1`, `libglib2.0-0` | required by OpenCV |
| pre-downloaded OCR/inpaint weights | no fetch needed at boot |

Version pins are fixed so the stack is reproducible, and the image is validated
by a **build-time smoke test** (it imports the OCR service and calls `/prepare`
on a synthetic page).

## Tags

- `0.1.0`, `latest`

Platform: `linux/amd64`.

## Services

| Port | Service |
|---|---|
| `8000` | LLM server (OpenAI-compatible `/v1`) |
| `8080` | OCR + inpainting service |

### Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `:8000/v1/models` | GET | readiness (returns the loaded model) |
| `:8000/v1/chat/completions` | POST | translation |
| `:8080/health` | GET | `{"ok": true, "detector": bool, "inpaint": "lama", "hf_mbps": int}` |
| `:8080/prepare` | POST (multipart `file`) | detect + OCR + erase; returns blocks and a cleaned page |

## Run

```bash
docker run --gpus all \
  -p 8000:8000 -p 8080:8080 \
  poynt2005/hanhua-gpu:0.1.0
```

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `SAKURA_MODEL` | a visual-novel-tuned 14B GGUF on HuggingFace | model to pull |
| `OLLAMA_CONTEXT_LENGTH` | `8192` | LLM context length |
| `OLLAMA_KEEP_ALIVE` | `30m` | how long to keep the model loaded |
| `PORT` | `8080` | OCR service port |
| `OCR_TOKEN` | empty | if set, `/ocr` and `/prepare` require `Authorization: Bearer <token>` |
| `INPAINT_DILATE` | `9` | mask dilation for text removal |
| `DEVICE` | auto (cuda if a GPU is visible, else cpu) | device for the detector |

## Startup behaviour

All model weights are baked in, so a normal boot makes **no network calls at all**.

1. Starts the OCR service first, so `/health` is reachable immediately.
2. Starts the LLM server.
3. If the model is already baked in (the default) it does nothing else — no
   HuggingFace probe, no download.
4. Only in the non-baked fallback: probes the model host's speed into
   `/root/hf_mbps.txt` (exposed as `hf_mbps` in `/health`) and pulls the model if
   the speed is above a threshold; otherwise it skips, so a controller can fail
   fast and switch machines instead of waiting.

The runtime is forced offline (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`) so the
baked weights load with zero network calls.

## Image size and the model

- `0.1.0` (~6.7 GB) does **not** contain the translation model; it is downloaded
  on first boot (depends on HuggingFace throughput).
- `0.2.2` / `latest` (~15.5 GB) bakes the model **and every OCR / inpainting
  weight** into fixed cache paths, and forces the runtime offline, so a boot makes
  no network calls at all. Prefer this tag.

To build your own with the model baked in, build the source Dockerfile with
`--build-arg BAKE_SAKURA=1`.

## Using with vast.ai

```bash
# rent + deploy with the hanhua CLI
python hanhua.py vast up --yes --image poynt2005/hanhua-gpu:0.2.2 --baked --write-config config.json
# translate, then tear down
python hanhua.py auto -i book.cbz --ocr http --profile sakura
python hanhua.py vast down
```

## License / third-party notice

This image bundles third-party tools and downloads third-party model weights;
each carries its own license. In particular:

- the recommended translation model family is released under a **non-commercial**
  license — do not use it commercially;
- **comic-text-detector** is vendored (a pinned commit of `dmMaze/comic-text-detector`)
  and is licensed **GPL-3.0**; its `LICENSE` file is kept inside the image at
  `/opt/comic-text-detector/LICENSE`.

Review each dependency's license before use.
