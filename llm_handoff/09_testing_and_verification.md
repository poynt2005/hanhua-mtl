# 09 — Testing and verification

## Offline self-test (no network, no key)

```bash
python hanhua.py selftest [-o OUTDIR] [--profile sakura|openai]
```

It builds a synthetic page, runs the whole pipeline with the mock provider, and
writes the output plus a `project.json`. Use it to confirm the pipeline and
packing work after any code change.

## Dry run on a real archive (no cost)

```bash
python hanhua.py auto -i book.cbz --ocr mock
```

Validates unpack → (mock) extract → translate → render → repack on the real
archive without any GPU or API cost. Then verify page count/order.

## Endpoint checks before a real run

```bash
python hanhua.py ping --profile sakura
```

A single-sentence request; expect a sane translation and HTTP 200. For the OCR
service, `GET /health` should return `{"ok": true, ...}` and a real
`POST /prepare` should return blocks and a cleaned image.

## Delivery verification (required)

Compare the input and output archives:

```python
import zipfile
ext = (".png", ".jpg", ".jpeg", ".webp")
a = [n for n in zipfile.ZipFile(src).namelist() if n.lower().endswith(ext)]
b = [n for n in zipfile.ZipFile(out).namelist() if n.lower().endswith(ext)]
assert a == b, "page names/order changed"
```

Requirements:
- same number of image entries,
- **same names in the same order**,
- spot-check one or two pages visually to confirm the target language is present.

## Useful mechanical checks

- **Dimensions preserved**: compare `Image.open(...).size` for a few pages. (A
  bug once changed dimensions because inpainting padding was not cropped back.)
- **Erase effectiveness**: compare dark-pixel counts inside the text boxes before
  and after erasing, with text **not** drawn (otherwise the new text inflates the
  count and the metric is meaningless).
- **File size sanity**: a large drop in JPEG size usually means the re-encode
  quality is wrong.

## Regression checklist after code changes

1. `python -m py_compile` on all modules.
2. `bash -n` on shell scripts.
3. `hanhua.py selftest` (both profiles).
4. `hanhua.py auto --ocr mock` on a small archive, then verify page order.
5. If the Dockerfile changed, rebuild (the build-time smoke test runs the service).

## What the build-time smoke test covers

`deploy/Dockerfile` ends with a step that imports the OCR service and calls
`/prepare` on a synthetic image, asserting the response has `blocks` and
`cleaned`. This catches the dependency problems (missing libGL, incompatible
`transformers`, missing multipart helper) at build time instead of on a paid
machine.
