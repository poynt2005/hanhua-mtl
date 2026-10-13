# hanhua

**One archive in, one translated archive out.**

`hanhua` is a CLI tool that takes a manga/comic archive (`.zip`, `.cbz`, or
`.epub`), translates the text into Traditional Chinese, erases the original
lettering with a neural inpainting model, typesets the translation back onto the
pages, and repackages everything in the original format — same page names, same
order.

It is built around a simple idea: **keep every stage separate and inspectable**.
An intermediate JSON holds each text box and its translation, so you can re-run
or hand-correct translation without redoing OCR, and re-run rendering without
re-translating.

> This is **machine translation**. Output is not human-reviewed. If you publish
> results, label them as machine translation.

---

## Motivation

Handling this by hand is six steps (source images → translate → proofread → clean
artwork → typeset → package), each with its own tool. `hanhua` wires the existing
tools together into one pipeline, adds a JSON round-trip so the work stays
inspectable, and enforces term/name consistency. It also packages the GPU side
into a container image and ships hard-won cost-safety rules for renting GPU time.

## What it delegates to (no wheels reinvented)

| Stage | Tool |
|---|---|
| Text detection | `comic-text-detector` (fallback: a general detector) |
| OCR | `manga-ocr` |
| Inpainting / erase | `LaMa` (`simple-lama-inpainting`) |
| Translation | `Sakura` (self-hosted, visual-novel tuned) or any OpenAI-compatible LLM |
| Simplified → Traditional | `OpenCC` |
| Typesetting | built in (horizontal and vertical CJK) |

## Goals

- **Same format out.** Same archive type, same page names, same order.
- **Model-based quality.** Neural erase with a text mask, manga-specialised OCR,
  a model tuned for this kind of dialogue.
- **Inspectable.** An intermediate JSON you can edit and re-run.
- **Consistent.** One source term → one canonical translation, verified before
  delivery.
- **Cheap and safe to run.** A prebuilt GPU image plus explicit cost guardrails.

## Non-goals

- Not a human-quality translation.
- Does not acquire source images.
- Not pixel-perfect layout (long vertical lines may need manual tweaks).

---

## Requirements

- Python 3.10+ (the core needs only `Pillow`).
- Optional, per backend: `opencc-python-reimplemented` (Traditional conversion),
  `simple-lama-inpainting` + `torch` (local LaMa), `manga-ocr` + `torch` (local
  OCR), `opencv-python` + `numpy` (OpenCV inpainting).
- For GPU work: a machine that can run the provided image (see below).

## Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

### One-shot (recommended)

```bash
python hanhua.py auto -i book.cbz --ocr http --profile sakura --inpaint lama
```

Produces `book_MTL_zhTW.cbz` and `book_MTL_zhTW.cbz.project.json`.

### Offline dry run (no cost)

```bash
python hanhua.py auto -i book.cbz --ocr mock
```

### Staged (when a human reviews translation)

```bash
python hanhua.py unpack    -i book.cbz -o raw/
python hanhua.py extract   -i raw/ -o project.json --ocr http
python hanhua.py terms     -p project.json -o glossary.json
#   review glossary.json (one source term -> one target)
python hanhua.py translate -p project.json -o project.zh.json --glossary glossary.json
python hanhua.py normalize -p project.zh.json --glossary glossary.json
python hanhua.py verify    -p project.zh.json --glossary glossary.json
python hanhua.py render    -p project.zh.json -i raw/ -o out/
python hanhua.py repack    -i book.cbz --rendered out/ -o book_MTL_zhTW.cbz
```

See `AGENTS.md` for the full command reference and `llm_handoff/` for the deep
documentation.

## GPU deployment (self-hosted models)

The GPU side runs two services on one machine:

- LLM server on port **8000** (OpenAI-compatible `/v1`)
- OCR + inpainting service on port **8080** (`/health`, `/prepare`)

Everything is baked into an image so a rented box boots ready in about a minute:

```bash
docker build -f deploy/Dockerfile -t <registry>/hanhua-gpu:1 deploy
docker push <registry>/hanhua-gpu:1
```

`.github/workflows/build-image.yml` builds and pushes automatically (it frees disk
on the runner first). To rent and deploy with the tool:

```bash
python hanhua.py vast up --yes --image <registry>/hanhua-gpu:1 --baked --write-config config.json
python hanhua.py auto -i book.cbz --ocr http --profile sakura
python hanhua.py vast down
```

**Before renting anything, read `deploy/VASTAI_PLAYBOOK.md`.** It contains hard
rules and a pitfall table paid for with real money.

## Configuration

Copy `config.example.json` to `config.json` and edit, or use environment
variables (`MTH_API_KEY`, `MTH_API_BASE`, ...). `config.json` is gitignored and
must never be committed.

## Repository hygiene

This is a public repository. It does **not** track and must not contain:

- any comic/manga archives or pipeline artifacts (source or translated text),
- credentials, API keys, tokens, or personal data,
- the local reference material under `llm_handoff_example/`.

`.gitignore` enforces this. See `llm_handoff/` for details.

## Documentation

| File | Contents |
|---|---|
| `AGENTS.md` | operating instructions for AI agents |
| `deploy/VASTAI_PLAYBOOK.md` | GPU rental hard rules and pitfalls |
| `skills/vastai-cost-safety/` | portable cost-safety skill |
| `llm_handoff/INDEX.md` | full handoff documentation (start here) |

## License

The pipeline code in this repository is provided as-is for personal use.
It delegates to third-party tools and models that carry their own licenses; check
each before use. In particular, the recommended translation model family is
released under a non-commercial license — do not use it commercially.

---

## Credits

This project was designed and built by an **AI coding agent**:

- **Model**: DeepSeek V4.1 Flash (`deepseek-v4.1-flash`), provider `opencode-go`
- **Harness**: OpenCode (agentic coding environment)

Human direction, requirements, review, and testing were provided by the
repository owner.
