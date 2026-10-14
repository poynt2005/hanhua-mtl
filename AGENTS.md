# AGENTS.md — instructions for AI agents working on this repo

`hanhua` is a CLI tool (`hanhua.py`) that translates a manga/comic archive into
Traditional Chinese and **repackages it in the original format**. Your job is
usually: **receive a `.zip` / `.cbz` / `.epub`, return a translated archive of
the same format.**

> ⛔ **Before touching vast.ai (renting GPUs, SSH, destroying), read
> `deploy/VASTAI_PLAYBOOK.md`.** It contains hard rules and a pitfall table paid
> for with real money (report budget before renting, 5-minute network check,
> $1 cap per task, always verify teardown). Money/safety rules in that file win.

The tool **does not reinvent wheels**; each stage delegates to an existing tool:

| Stage | Delegated to |
|---|---|
| Text detection | `comic-text-detector` (or the OCR service fallback) |
| OCR | `manga-ocr` |
| Inpainting (erase) | `LaMa` (`simple-lama-inpainting`) |
| Translation | `Sakura` (visual-novel tuned) or any OpenAI-compatible LLM |
| Simplified→Traditional | `OpenCC` |

---

## 0. Hard rules

1. **Never modify the input file.** Always write to a new path.
2. **Never rename pages or drop pages**: the output archive's entries must match
   the input's names and order exactly.
3. **Verify before delivering** (see §5). Do not report success otherwise.
4. This is **machine translation**; state "not human-reviewed" when reporting.
5. **Term consistency is a hard requirement**: one source term → exactly one
   canonical translation, managed by `glossary.json` (§4.5). If inconsistent,
   normalise first; only deliver when `verify` passes.
6. Use only the commands in this file. Do not download models or rewrite code
   unless the user asks.

---

## 1. Environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` installs only `Pillow` (the core). Optional extras:

```
pip install opencc-python-reimplemented   # simplified -> traditional
pip install simple-lama-inpainting        # inpaint=lama (local; needs torch)
pip install manga-ocr                     # local OCR (needs torch)
pip install opencv-python numpy           # inpaint=opencv / textmask
```

> On the GPU box everything (OCR + inpainting + LLM) runs in a prebuilt image —
> see `deploy/Dockerfile` and `llm_handoff/04_gpu_deployment.md`.

---

## 2. Configuration

Copy `config.example.json` to `config.json`, or use environment variables
(`MTH_API_KEY`, ...). Key fields:

| Field | Meaning |
|---|---|
| `profile` | `openai` (any OpenAI-compatible LLM) or `sakura` (self-hosted) |
| `api_base` / `api_key` / `text_model` | translation endpoint |
| `ocr_url` / `ocr_prepare_url` | self-hosted OCR+inpaint service (`--ocr http`) |
| `target_lang` | default `zh-TW` |
| `convert_tw` | convert Sakura's simplified output to Traditional (needs OpenCC) |
| `inpaint` | `textmask` (recommended) / `fill` / `blur` / `opencv` / `lama` |

---

## 3. Primary command: one archive in, one archive out

```powershell
python hanhua.py auto -i <input.zip|cbz|epub> [-o <output>] `
    --ocr <backend> --profile <profile> [--inpaint lama] [--glossary glossary.json]
```

Produces:
- `<output>`: translated archive, same format, same page order.
  Default name: `<original-stem>_MTL_zhTW<ext>`.
- `<output>.project.json`: per-block source/translation (for review).

### Backend decision tree

```
Is the translation endpoint a self-hosted Sakura?  -> --profile sakura
  otherwise (cloud key available)                  -> --profile openai
OCR:
  config.ocr_url set                               -> --ocr http
  local manga-ocr/torch installed                  -> --ocr manga_ocr
  validate the pipeline only, offline              -> --ocr mock
```

`--ocr` choices: `vision` `http` `manga_ocr` `opencv` `sidecar` `mock`
`--profile` choices: `openai` `sakura`
`--inpaint` choices: `textmask` `fill` `blur` `opencv` `lama`

---

## 4. Recommended order

```powershell
# 4.1 test the translation endpoint first
python hanhua.py ping --profile sakura

# 4.2 dry run to validate packing (fake data, offline)
python hanhua.py auto -i book.cbz --ocr mock

# 4.3 real run
python hanhua.py auto -i book.cbz --ocr http --profile sakura --inpaint lama
```

Staged mode (when a human reviews translation):

```powershell
python hanhua.py unpack    -i book.cbz -o raw/
python hanhua.py extract   -i raw/ -o project.json --ocr http
python hanhua.py translate -p project.json -o project.zh.json --profile sakura
python hanhua.py render    -p project.zh.json -i raw/ -o out/
python hanhua.py repack    -i book.cbz --rendered out/ -o book_MTL_zhTW.cbz
```

---

## 4.5 Quality first: term / name consistency

```powershell
python hanhua.py terms     -p project.json -o glossary.json --min-count 2
#   <- YOU (the agent) review the zh column; one source term -> one target
python hanhua.py translate -p project.json -o project.zh.json --glossary glossary.json
python hanhua.py normalize -p project.zh.json --glossary glossary.json
python hanhua.py verify    -p project.zh.json --glossary glossary.json   # exit 0 required
```

Rules: `terms` is a proposal; review it. `normalize` rewrites deviations.
`verify` non-zero exit = do not deliver.

---

## 5. Verification before delivery (required)

```powershell
python - <<'PY'
import sys, zipfile
src, out = sys.argv[1], sys.argv[2]
ext = (".png", ".jpg", ".jpeg", ".webp")
a = [n for n in zipfile.ZipFile(src).namelist() if n.lower().endswith(ext)]
b = [n for n in zipfile.ZipFile(out).namelist() if n.lower().endswith(ext)]
print("src pages:", len(a), "out pages:", len(b), "same order:", a == b)
PY
```

Requirement: same page count, same order, and spot-check 1–2 pages for Chinese.

---

## 6. Error handling

| Symptom | Cause | Fix |
|---|---|---|
| `missing API key` | no `api_key` / `MTH_API_KEY` | set it, or use a self-hosted endpoint |
| network error / timeout | endpoint down or wrong port | check the port mapping; raise `--timeout` |
| refusal / empty translation | cloud model content policy | use a self-hosted `--profile sakura` |
| `translation response had no usable entries` | model did not follow the format | retry; or switch model |
| line count mismatch | model merged/split lines | Sakura mode falls back to single-line; lower `translate_max_lines` |
| `No CJK font found` | font missing | set `config.font` or `MTH_FONT` |
| `--ocr http needs config ocr_url` | no OCR endpoint | set `ocr_url` or use `--ocr manga_ocr` |
| residual text on pages | erase incomplete | `--inpaint lama`; raise `padding`/`dilate` |
| text overflows the bubble | box small / text large | normal for long vertical lines; edit `translation` and `render` |
| `verify` reports inconsistency | same source, multiple translations | `normalize --glossary`, or fix the glossary |

---

## 7. Cost & time (one ~180-page volume)

| Item | Notes |
|---|---|
| Translation (self-hosted) | GPU rental only, no token cost |
| Translation (cloud) | a few cents |
| OCR / inpainting | free (local or self-hosted) |
| One full run | minutes on GPU; tens of minutes on CPU |

---

## 8. Self-hosted backend (vast.ai)

**Pick cost-effective machines.** Prefer RTX 3090 24GB (lowest unit price, 24GB
is enough for 14B + OCR on one box); use 4090 for speed; budget option is a
16GB card with 7B. Run **one** instance for both services (8000 = LLM,
8080 = OCR), sort by `$/hr`, set a max price, and destroy when done.

Full steps in `deploy/VASTAI_PLAYBOOK.md`:
- `deploy/Dockerfile` + `deploy/container_start.sh`: prebuilt image with the
  whole stack; boot ready in ~1 minute.
- `hanhua.py vast up --yes --image <tag> --baked`: rent + deploy + return endpoints.
- `hanhua.py vast down`: destroy everything.

When the user gives you `api_base` / `ocr_url`, put them in `config.json` and use
`--profile sakura --ocr http`.

**⛔ Key hard rules (see the playbook):**

- No renting without explicit user consent; report the budget first.
- **$1 cap per task**; first line of every report is the money.
- **One machine at a time.** `vast up` refuses to rent while one of ours is
  running/loading. **Never write a script that can rent** - rental only happens
  through `hanhua.py vast up`. Run `deploy/vast_watchdog.py` while a run is in
  flight; it destroys any extra instance we own (never the user's others).
- **A failed run KEEPS the machine.** It is the debugging environment. Destroy
  only when the work is verified complete, or the user says so.
- **Never send a long silent request.** Stream LLM responses and retry transient
  drops in both the LLM and the OCR client; a non-streaming request is silent
  during generation and the host's port forward drops it after ~15 s.
- Prefer hosts that already hold the image (Docker Hub speed is per host); if a
  pull will not finish in ~15 minutes, destroy and switch; blacklist slow hosts.
- Run `deploy/net_check.sh` within 5 minutes of boot; on `NET_BAD` destroy and
  switch machines — do not debug a bad host.
- Three gates before a batch: `/v1/models` returns a model → single-sentence test
  200 → no stale process.
- Judge success by **content** (non-empty output) and by `tools/verify_batch.py`
  (page count, page order, image dimensions, translation coverage), not line
  counts.
- Never `scp` model weights or big wheels to the remote; download on the remote.
- Before destroying, pull artifacts back and verify; after destroying, confirm
  `show instances` is empty.
- API keys live only in environment variables; never write them to files.

> The measured cost of doing this right: **~15 cents per 250-page volume**
> (~15 s/page). See `llm_handoff/13_operating_manual.md` for the runbook and
> `deploy/VASTAI_PLAYBOOK.md` for the pitfall table (V43-V57 are this project's
> own, paid for in real money).
>
> ⚠️ **Before you change anything, read `llm_handoff/14_thrashing_postmortem.md`.**
> It documents how this project repeatedly broke its own fixes by guessing the
> cause and testing on a paid machine, destroying the machine that held the
> evidence, and editing controllers without re-checking their call sites. Every
> dollar of waste came from that loop. The rule is: **reproduce for free,
> instrument, change ONE thing, verify in isolation, and only then spend.**

---

## 9. One-line summary

```
1) selftest + auto --ocr mock           (free, local: validates packing)
2) tools/rent_and_run.py --yes --image <tag> --tag <tag> --run <pages>...
                                        (rent one machine, MEASURE its pull,
                                         gate, run, verify, tear down)
3) tools/verify_batch.py --dir .        (page count/order/dimensions/coverage)
4) tools/audit_sheets.py --dir .        (contact sheets: a human should glance)
5) tools/collect.py --dir . --dest ComicsZHTW
6) report: money first, output paths, and "machine translation, not reviewed"
```

`rent_and_run.py` is the entry point; it never leaves a machine behind and it
refuses to rent while one is already recorded. Read
`llm_handoff/13_operating_manual.md` (runbook) and
`llm_handoff/14_thrashing_postmortem.md` (how not to waste money) first.
