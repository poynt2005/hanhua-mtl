# 12 — Network / HuggingFace audit (every function that can open a connection)

Purpose: after a pilot failure caused by a model trying to reach HuggingFace at
runtime, this document answers one question exhaustively — **which functions can
open a network connection, and which of them touch HuggingFace?** It is the
reference for keeping the runtime offline.

## Summary — the whole HuggingFace surface

There is exactly **one HuggingFace model load** in the codebase, plus two
non-model HuggingFace touchpoints. Everything else downloads from GitHub
releases, or is local.

| # | Where | What | HuggingFace? | Runs when |
|---|---|---|---|---|
| 1 | `ocr_server.get_mocr()` (box) and `mth/ocr._load_manga_ocr()` (client) | `manga_ocr.MangaOcr()` → `transformers …from_pretrained("kha-white/manga-ocr-base")` | **YES** | every `/prepare` on the box; only local OCR backends on the client |
| 2 | `container_start.sh` HF probe | `curl https://huggingface.co/…sakura….gguf` (50 MB range, 25 s cap) | **YES** (not a model) | boot — now only in the non-baked fallback |
| 3 | `container_start.sh` / `Dockerfile` model pull | `ollama pull hf.co/SakuraLLM/…` | **YES** (not a model) | build (baked) or boot fallback only |
| 4 | `easyocr.Reader(...)` | model files from `github.com/JaidedAI/EasyOCR/releases` | no (GitHub) | only if the local cache is missing/corrupt |
| 5 | `simple_lama_inpainting.SimpleLama()` | `big-lama.pt` from `github.com/enesmsahin/simple-lama-inpainting/releases` | no (GitHub) | only if the local cache is missing |
| 6 | `inference.TextDetector(model_path=…)` from the **vendored** checkout | loads a **local** `.pt` | no | lazy, on first detect |

Only #1 is a *model* load on HuggingFace, and it is the one that broke the pilot.
#2 and #3 are the Sakura GGUF; #3 is eliminated by baking the model and #2 is now
skipped entirely on a baked image.

## Why #1 fails even with a warm cache

`transformers` / `huggingface_hub` issue a Hub *freshness* request on every
`from_pretrained`, even when the weights are already in the local cache. On a
datacenter host where `huggingface.co` is unreachable the call retries with
backoff, so the first OCR request hangs and the `/prepare` HTTP call fails
(`Remote end closed connection without response`). Setting
`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` makes it read the cache with **zero**
network calls. See bug **G9** in `06_bugs_and_fixes.md`.

## Per-function audit

### `deploy/ocr_server.py` (runs on the rented GPU box)

| Function | Network? | HuggingFace? | Notes |
|---|---|---|---|
| `_pick_device()` | no | no | picks cuda only if a GPU is visible |
| `get_reader()` | only if cache missing | no | `easyocr.Reader(["ja"])`, GitHub releases → `$EASYOCR_MODULE_PATH` |
| `get_mocr()` | **yes on load** | **YES** | `MangaOcr()` → `$HF_HOME`; needs offline flags |
| `_import_text_detector()` | no | no | imports `inference` from the vendored dir (appended to `sys.path`) |
| `_detector_weights()` | no | no | local path resolution only |
| `get_detector()` | no | no | loads `comictextdetector.pt` from the vendored dir |
| `get_lama()` | only if cache missing | no | `big-lama.pt` → `$TORCH_HOME/hub/checkpoints` |
| `detect()` | via get_detector / get_reader | indirect | no direct connection |
| `build_mask()` | no | no | OpenCV only |
| `inpaint()` | via get_lama | no | |
| `read_blocks()` | via get_mocr | **YES** | this is the call that broke the pilot |
| `_auth()` | no | no | |
| `_hf_mbps()` | no | no | reads a local file |
| `health()` | no (local files) | no | loads the detector lazily; no download |
| `ocr()` | via detect + read_blocks | indirect | |
| `prepare()` | via detect + read_blocks + get_lama | **YES** | the client's hot path |

### `deploy/container_start.sh`

| Step | HuggingFace? | Now |
|---|---|---|
| HF speed probe (`curl` the GGUF) | **yes** (non-model) | **skipped when the model is baked** |
| `ollama pull hf.co/…` | **yes** (non-model) | build-time only (baked) |
| `ollama list` / `ollama serve` | no | local |
| start `ocr_server.py` | no | local |

On a baked image a normal boot now makes **no network calls at all**.

### `deploy/Dockerfile` (build time only)

| Step | HuggingFace? | Note |
|---|---|---|
| `pip install` (transformers, manga-ocr, easyocr, LaMa, …) | no | PyPI |
| `git fetch` comic-text-detector at a pinned commit | no | GitHub |
| `pip install pyclipper shapely torchsummary …` | no | PyPI |
| pre-download manga-ocr / easyocr / LaMa | **yes** (manga-ocr only) | written into the fixed cache paths |
| comic-text-detector weights | no | GitHub release |
| `ollama pull hf.co/…` (BAKE_SAKURA=1) | **yes** (non-model) | baked once, at build |
| smoke test | no | runs with **every outbound socket blocked** |

### `mth/ocr.py` (client)

| Function | Network? | HuggingFace? | Notes |
|---|---|---|---|
| `_load_manga_ocr()` / `recognize_manga_ocr()` | **yes on load** | **YES** | only used by local backends (`manga_ocr`, `opencv`) |
| `_vision()` | yes | no | the configured OpenAI-compatible endpoint |
| `_items_to_blocks()` / `_sidecar()` / `_prepare_url()` | no | no | |
| `_http()` | yes | no | POSTs to `ocr_url` (the box) |
| `_text_detector_class()` / `_detector_weights()` / `detect_boxes()` | no | no | vendored dir + a local `.pt` |
| `_opencv()` | via manga-ocr | **indirect** | cv2 boxes + manga-ocr text |
| `_manga_ocr()` | via manga-ocr | **indirect** | vendored detector boxes + manga-ocr text |
| `_mock()` / `extract_page()` | no | no | |

### `mth/inpaint.py` (client)

| Function | Network? | HuggingFace? |
|---|---|---|
| `_lama_method()` | only if cache missing | no (GitHub) |
| `_opencv_method()`, `_textmask_method()`, `_fill_method()`, `_blur_method()`, `_clamp()`, `_expand()`, `_border_mean()`, `erase()` | no | no |

### `mth/providers.py` (client)

| Function | Network? | HuggingFace? |
|---|---|---|
| `OpenAICompat.chat()` / `.vision()` | yes | no (endpoint is whatever `api_base` is) |
| `_data_uri()`, `_extract_json()`, `build_client()`, `Mock.*`, `vision_extract()`, `_translate_openai()`, `_sakura_messages()`, `_split_lines()`, `_translate_sakura()`, `translate_batch()`, `ping()` | no | no |

### `mth/vast.py` (client)

| Function | Network? | HuggingFace? |
|---|---|---|
| `bootstrap_script()` (non-baked onstart) | yes | **yes** — the same GGUF speed probe |
| `wait_http()` | yes | no — polls the box's `/health` and `/v1/models` |
| `create_instance()`, `wait_running()`, `instance_info()`, listing/destroy | yes | no (console.vast.ai) |

### Everything else

`mth/archive.py`, `mth/pipeline.py`, `mth/render.py`, `mth/terms.py`,
`mth/config.py`, `mth/schema.py`, `hanhua.py` — no network, no HuggingFace.

## What was changed because of this audit

1. **Fixed cache paths** (`HF_HOME`, `TORCH_HOME`, `EASYOCR_MODULE_PATH`) so the
   build writes and the runtime reads the *same* location.
2. **Every used weight is baked**: manga-ocr (HF), EasyOCR, LaMa, and the
   comic-text-detector weights.
3. **comic-text-detector is vendored** at a pinned commit (it cannot be
   pip-installed; see the next section), with the small runtime deps it needs and
   the training-only `wandb` import made optional.
4. **Runtime is offline** (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`), set
   after the download steps.
5. **No HF probe on a baked boot** — the start script checks for the baked model
   first, so a same-image boot makes no network calls at all.
6. **The build smoke test blocks every outbound socket**, so anything that still
   needs the network fails the build instead of a GPU box.
7. `get_detector()` no longer hard-codes `device="cuda"`, and `detect()` now uses
   the detector's real return value (`mask, mask_refined, block_list`).

## Why comic-text-detector is vendored, not installed

`pip install git+https://github.com/dmMaze/comic-text-detector.git` **cannot
work**, and it never did — which is why every machine reported `detector: false`:

- the repository has **no `setup.py` and no `pyproject.toml`**, and **no tags**,
  so pip reports "does not appear to be a Python project";
- it is **not on PyPI**;
- its modules use **top-level imports** (`from basemodel import …`), so it is only
  usable with the checkout directory on `sys.path` — not as `comic_text_detector.*`;
- `utils/db_utils.py` additionally needs `pyclipper` and `shapely`.

The image therefore pins a commit, fetches it into `/opt/comic-text-detector`,
deletes the training-only scripts, installs `pyclipper`, `shapely`, `torchsummary`
(and `requests`, `tqdm`), and makes the single `import wandb` in
`utils/general.py` optional. Both `deploy/ocr_server.py` and `mth/ocr.py` append
that directory to `sys.path` (appended, never prepended, so the generic top-level
names `utils`/`models` cannot shadow site-packages).

> **Licence note:** comic-text-detector is **GPL-3.0** (unlike manga-ocr, EasyOCR
> and LaMa, which are Apache-2.0). Its `LICENSE` file is kept inside the image.

## The fully-offline proof

The build smoke test patches `socket.socket.connect` to raise on any non-loopback
address before it does anything else, then:

1. calls `/health` and asserts `detector is True`;
2. runs `detect()` and asserts it returns a text mask;
3. runs `inpaint()` (LaMa) and `get_mocr()` (manga-ocr);
4. calls `/prepare` end to end.

If any of those reached out to HuggingFace, GitHub or anywhere else, the build
fails. That is the machine-checked guarantee that the container needs no network
at runtime.

## Residual, accepted network use

- `easyocr` / `simple-lama-inpainting` still *could* reach GitHub **only** if a
  baked file were missing or its checksum mismatched. With the weights baked this
  does not happen.
- The client still reaches the box (`/prepare`, `/v1`) and `console.vast.ai`.
- `api_base` is whatever the user configured; if it points at HuggingFace, that
  is the user's choice, not this code.
