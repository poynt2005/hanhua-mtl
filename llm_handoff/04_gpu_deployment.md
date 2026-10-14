# 04 — GPU deployment

## The stack that runs on the GPU box

| Component | Role | Port |
|---|---|---|
| LLM server (ollama) | translation model (OpenAI-compatible `/v1`) | 8000 |
| OCR+inpaint service (`ocr_server.py`) | detection, OCR, text removal | 8080 |

The client (`hanhua.py`) only speaks HTTP to these two, so **SSH is not needed
for the happy path**.

## The prebuilt image (`deploy/Dockerfile`)

Base: a CUDA + PyTorch runtime image (PyTorch is preinstalled, so CUDA wheels are
not re-downloaded). The image adds:

- system libs: `libgl1`, `libglib2.0-0` (OpenCV needs libGL), `git`, `curl`, `zstd`
- pinned Python deps: a compatible `transformers`, `manga-ocr`, `easyocr`,
  `fastapi`, `uvicorn`, `python-multipart`, `pillow`, `numpy<2`,
  `simple-lama-inpainting`, `httpx`
- a constraints file that stops pip from replacing CUDA PyTorch with a CPU wheel
- the detector installed from its code host (it is **not on the package index**)
- ollama
- **every model weight baked in** into fixed cache paths: manga-ocr
  (HuggingFace), EasyOCR, LaMa, and comic-text-detector — boots never fetch a
  model
- **comic-text-detector vendored** at a pinned commit (it has no package metadata,
  so `pip install` is impossible), plus its `pyclipper` / `shapely` /
  `torchsummary` deps and a patched-out `wandb` import
- the runtime **forced offline** (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`)
- a **build-time smoke test that blocks every outbound socket**, then asserts the
  detector, LaMa and manga-ocr all load and that `/prepare` works — a
  machine-checked "fully offline" guarantee

Optional build arg `BAKE_SAKURA=1` also bakes the translation model (~9 GB) into
the image, removing the boot-time download.

Build (no GPU required):

```bash
docker build -f deploy/Dockerfile -t <registry>/hanhua-gpu:1 deploy
docker push <registry>/hanhua-gpu:1
```

CI: `.github/workflows/build-image.yml` builds and pushes automatically (it frees
disk on the runner first, because the base image plus deps is large).

## Container start (`deploy/container_start.sh`)

1. Start the OCR service **first**, so `/health` is reachable immediately.
2. Start the LLM server.
3. If the model is already baked in (the normal case) it does **nothing else** —
   no HuggingFace probe, no download.
4. Only in the non-baked fallback: probe the model host's speed, write it to a file
   (exposed by `/health` as `hf_mbps`), and pull the model if the speed is above a
   threshold; otherwise skip so the controller can fail fast.

## Readiness detection (client side)

`mth/vast.py` polls the two HTTP endpoints and declares readiness only when:

- the LLM endpoint returns a **non-empty model list** (a server with no model
  still answers 200 with an empty list — that must not count as ready), and
- the OCR `/health` responds.

`hf_mbps` is reported by `/health` for information only; readiness never gates on
HuggingFace speed, because the models are baked into the image.

## The `vast` command

```bash
python hanhua.py vast up --yes --image <registry>/hanhua-gpu:1 --baked \
       --attempts 4 --max-price 0.20 --write-config config.json
```

`up` does:
1. destroy leftover instances **labelled by this tool** (never the user's others),
2. search offers (cheapest inside a reliability band, under a max price),
3. create one instance, opening ports 8000/8080,
4. wait for it to run, then poll the HTTP endpoints,
5. on failure: destroy and try the next offer (skipping hosts that already failed),
6. on success: return `{instance_id, api_base, ocr_url, text_model}` and optionally
   write them into `config.json`.

`down` destroys all instances and verifies the list is empty.

## Boot modes

| Mode | When to use | Boot time |
|---|---|---|
| **Baked image** (`--image ... --baked`) | normal operation | ~1 minute |
| Generic image + onstart install | no image available | 8–15 minutes, fragile |

The generic mode embeds the install as a base64 onstart script; it exists as a
fallback but the baked image is strongly preferred.

## Current state (important)

- The image definition, container start script, and CI workflow are written and
  syntax-checked.
- **`v0.2.2` is the current image**: it bakes the translation model **and every
  OCR / inpainting weight** into fixed cache paths, and forces the runtime offline,
  so a boot makes no network calls at all. The build smoke test runs offline and
  fails if any cache is incomplete.
- History: `v0.2.0` baked the model but still let `transformers` phone HuggingFace
  on every load (see `06_bugs_and_fixes.md` G9); `v0.1.0` (~6.7 GB) fetched the
  model at boot and was HF-dependent.
  Reference: `docker.io/<DOCKERUSR>/hanhua-gpu:0.2.2` (and `:latest`).

## Endpoint reference

| Endpoint | Method | Purpose |
|---|---|---|
| `http://<host>:8000/v1/models` | GET | LLM readiness (non-empty list) |
| `http://<host>:8000/v1/chat/completions` | POST | translation |
| `http://<host>:8080/health` | GET | OCR readiness + `hf_mbps` |
| `http://<host>:8080/prepare` | POST (multipart `file`) | detect + OCR + erase |
