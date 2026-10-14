# 02 — Architecture

## Pipeline stages

```
archive (.zip/.cbz/.epub)
        │
        ▼  unpack            extract image entries to a temp dir, keep a name map
   raw images
        │
        ▼  extract           detect text regions + OCR  ->  project.json
   project.json (per page: boxes, direction, source text)
        │
        ▼  translate         batch the source text to an LLM -> translations
   project.zh.json
        │
        ▼  normalize/verify  enforce glossary consistency
        │
        ▼  render            erase source text (inpainting) + typeset target text
   rendered images
        │
        ▼  repack            rebuild the archive, same names and order
archive_MTL_zhTW (.zip/.cbz/.epub)  +  .project.json sidecar
```

Stages are separate on purpose: the JSON round-trip means translation can be
re-run, reviewed, or corrected without repeating OCR or inpainting.

## Modules (`mth/`)

| Module | Responsibility |
|---|---|
| `config.py` | load `config.json` + environment overrides; locate a CJK font |
| `schema.py` | the project JSON structure (pages, blocks) |
| `archive.py` | unpack/repack archives; preserve entry names and order; natural sort |
| `providers.py` | OpenAI-compatible chat client (text + vision) that **streams** responses (SSE) and retries transient drops; translation profiles; a mock |
| `ocr.py` | detection+OCR backends: `http`, `manga_ocr`, `opencv`, `sidecar`, `vision`, `mock`; retries the (non-streaming) `/prepare` call |
| `inpaint.py` | text removal: `textmask` (OpenCV text mask), `fill`, `blur`, `opencv`, `lama` |
| `render.py` | typesetting: horizontal and vertical CJK, auto font sizing, stroke, contrast |
| `pipeline.py` | orchestration: extract / translate / render; JPEG quality preservation |
| `terms.py` | term extraction, glossary I/O, normalisation, verification |
| `vast.py` | vast.ai lifecycle: search, rent+deploy, readiness, destroy; the **single-instance invariant** (refuses to rent while one of ours is running/loading) |

Entry point: `hanhua.py` (argparse subcommands).

## The OCR+inpaint service (`deploy/ocr_server.py`)

A small FastAPI app that runs on the GPU box:

- `GET /health` → `{"ok": true, "detector": <bool>, "inpaint": "lama", "hf_mbps": <int>}`
- `POST /ocr` → `{"blocks": [...]}` (compatibility)
- `POST /prepare` → `{"blocks": [...], "cleaned": "<base64 PNG>"}`

Design points:
- Detection prefers a segmentation-capable detector; otherwise it falls back to a
  general text detector.
- Detection is **adaptive**: a plain pass first, then rotated (vertical text) and
  inverted (white-on-dark) passes when the plain pass finds suspiciously little
  text; rotated boxes are mapped back to the original coordinates and deduped by
  IoU. It is a safety net, not a fix (see `06_bugs_and_fixes.md` G17).
- `GET /logs` exposes the service log over HTTP, so a live machine can be
  diagnosed without SSH.
- The inpaint mask is **text-based** (Otsu inside each box, dilated) so only text
  strokes are erased — using whole rectangles would redraw bubble borders and
  artwork inside the box.
- LaMa pads to a multiple of 8; the service **crops back** to the original size.
- Failures in inpainting degrade gracefully (blocks are still returned).

## Client side

`mth/ocr.py` calls `/prepare`, saves the returned cleaned page as
`<page>.clean.png`, and returns the blocks. `mth/pipeline.py` uses the cleaned
page as the render base when present (skipping local erase), so the erase quality
comes from the model, not from a heuristic.

## Resilience (the things that used to abort a batch)

- **Every LLM request streams.** A non-streaming completion is silent while the
  model generates, and the GPU host's port forward drops the idle socket after
  ~15 s (measured: **4/6 non-streaming requests survived, 6/6 streaming**). This
  was the root cause of every aborted batch.
- The OCR `/prepare` call cannot stream, so the client **retries** it instead.
  Both paths are bounded by `config.retries`.
- Readiness requires a **non-empty model list**, not just HTTP 200.
- **One machine at a time**: `up` refuses while an instance of ours is running or
  loading, and `deploy/vast_watchdog.py` destroys any extra instance we own while
  a run is in flight.

## Project JSON schema

```json
{
  "version": 1,
  "meta": { "chapter": "", "source_lang": "ja", "target_lang": "zh-TW" },
  "glossary": { "<source term>": "<canonical translation>" },
  "pages": [
    {
      "page": 1,
      "image": "00001.jpg",
      "width": 1275,
      "height": 1913,
      "blocks": [
        {
          "id": "p001-b00",
          "bbox": [820, 140, 1030, 430],
          "direction": "vertical",
          "type": "speech",
          "ocr": "<source text>",
          "translation": ""
        }
      ]
    }
  ]
}
```

Key rules:
- Block ids are stable (`p<page>-b<index>`); all joins are by id.
- Coordinates never go to the LLM; only `ocr` text does.
- `type: "sfx"` blocks are kept as-is by default (`keep_sfx`).

## Rendering details

- Horizontal and vertical CJK layout; font size is chosen to fit the box.
- Text colour is picked for contrast against the (erased) background.
- Output JPEGs are saved with the **source image's quality** and
  `subsampling=0`, so quality is preserved.
- Output format follows the source extension when `keep_ext` is set (used by
  `auto`/`book`).

## Why an image for the GPU side

Installing torch + OCR + inpainting + the LLM server from scratch on every boot
is fragile and was the main source of wasted money. The image bakes the whole
stack, pins versions, pre-downloads weights, and runs a **build-time smoke test**,
so dependency problems surface during the free build instead of on a paid box.
