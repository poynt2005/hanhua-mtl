# 13 — Operating manual: how to translate a batch without wasting money

This is the runbook that actually worked (5 volumes, ~1,850 pages, one machine,
**under $1 total**). It assumes you have read `deploy/VASTAI_PLAYBOOK.md` and
`llm_handoff/12_hf_network_audit.md`. Every rule here was paid for.

---

## 1. The measured cost model (use these numbers to plan)

| Quantity | Measured value |
|---|---|
| Machine actually used | Tesla V100 32 GB @ **$0.153/hr** (cheapest 24 GB+ offer was a 3090 @ $0.170) |
| OCR + inpaint stage | **~6.9 s/page** (detector + manga-ocr + LaMa) |
| Translation stage | **~8.3 s/page** (Sakura 14B Q4_K_M, streaming) |
| **Whole pipeline** | **~15 s/page** (≈ 60 min for a 240-page volume) |
| **Cost per volume** | **~$0.15** (≈ 60 min × $0.153) |
| Model load / boot | ~1 min on a host with warm image layers; ~3 min `running` |
| 5 volumes end to end | ~3.5 h wall clock, **~$0.70** |

Rule of thumb: **a 250-page volume costs about 15 cents.** Anything much above
that means something is wrong (idle time, a slow host, or a stall).

Quantisation and model choice. **The image ships `sakura-qwen3-q6k`**: the
Sakura-14B-**Qwen3**-v1.5 release at **Q6_K** (12.12 GB), imported from its GGUF
with a Modelfile, running with a **16384-token context**.

Why this exact combination (all evidence-based):

- **Q6_K over Q4_K_M**: llama.cpp's perplexity table gives Q6_K **+0.0044** vs
  Q4_K_M **+0.0535**; r/LocalLLaMA calls Q6_K "almost identical to f16"; and the
  VRAM-headroom rule says "model under 60% of VRAM → use Q6_K". Q6_K needs
  ~16 GB, so it still fits a 24 GB card - the only cost is ~25% slower generation.
- **The Qwen3 base over GalTransl-v3.8**: same 14B size, same Q6_K size, same
  VRAM/speed class, newer base (released 2026-01).
- **16384 context** (up from 8192): Sakura's own guide runs 32768; 16k keeps the
  KV cache within a 24 GB card while giving the translator far more context for
  cross-page consistency.

| Quant | GGUF | VRAM needed | GPU class | Cost / 250-page volume | Note |
|---|---|---|---|---|---|
| IQ4_XS | 8.19 GB | ~12 GB | 24 GB | ~$0.13 | smallest, marginal vs Q4_K_M |
| Q4_K_M | 8.99 GB | ~13 GB | 24 GB | ~$0.14 | the old default; quality judged unsatisfactory |
| Q5_K_S | 10.27 GB | ~14 GB | 24 GB | ~$0.15 | community calls the `_S` mix weak |
| **Q6_K (current)** | **12.12 GB** | **~16 GB** | **24 GB** | **~$0.18** | "almost identical to f16" |
| bf16 (not published) | ~28 GB | ~33 GB | 40–80 GB | ~$0.60–0.85 | 4–6× the cost for a small gain |

Three implementation traps, all paid for:

1. **`ollama pull ...:Q6_K` cannot resolve** for these repos - the quant tag only
   matches when a *filename* contains it (for GalTransl the unsuffixed GGUF was
   Q6_K; for the Qwen3 release the file is `...-q6k.gguf`). Import the GGUF with
   a Modelfile (`FROM /root/<name>.gguf` + `ollama create <name>`) instead of
   guessing tags.
2. Importing the file needs **~2× its size in free disk** during the build (the
   GGUF plus the ollama blob), so `rm` the GGUF in the same layer.
3. **A new model version usually needs a new system prompt.** The Qwen3 release
   recommends different wording from the GalTransl line; shipping the model with
   the old prompt is a silent quality regression. Copy the prompt from the model
   card into `mth/providers.py`.

Before any batch, run the pre-flight gate: it translates three lines and requires
exactly three non-empty Chinese lines with **no reasoning/template noise**
(` thinking`, `<|im_start|>`, …) - which is what a wrong chat template or a
reasoning model would produce.

---

## 2. The procedure that works

```powershell
# 0) environment
py -m venv .venv ; .\.venv\Scripts\Activate.ps1 ; pip install -r requirements.txt
$env:VAST_API_KEY = "<key>"          # env only, never a file

# 1) free validation, no GPU
python hanhua.py selftest
python hanhua.py auto -i sample.zip --ocr mock      # validates packing

# 2) the whole remote flow in one command: rent ONE machine -> measure its pull
#    (a bad host is destroyed + blacklisted automatically) -> gate -> run the
#    volumes -> verify each -> tear the machine down
python tools/rent_and_run.py --yes --image poynt2005/hanhua-gpu:<tag> --tag <tag> `
       --config config.json --dir . --run 250 237

# 3) verify and deliver
python tools/verify_batch.py --dir .            # page count, order, dimensions, coverage
python tools/audit_sheets.py --dir .            # contact sheets for a human to glance at
python tools/collect.py --dir . --dest ComicsZHTW
```

`rent_and_run.py` refuses to rent without `--yes` and refuses to rent while one
machine is already recorded in `vast_current.txt`. If you prefer to drive the
steps yourself, the equivalents are `hanhua.py vast up --baked` (rent) →
`tools/pull_watch.py` (measure the pull) → `tools/gate.py` (prove the model
works) → `tools/run_batch.py` (run + verify + tear down). **Never skip
`pull_watch`**: the advertised bandwidth filter is not evidence, and a slow host
cannot be told from a stuck one by the status message.

During a run, keep a watchdog alive so a second machine can never start:

```powershell
$env:WATCH_INTERVAL = 45
python deploy/vast_watchdog.py
```

---

## 3. Why it works now (the improvements, in order of impact)
1. **Bake the entire environment into an image** (`deploy/Dockerfile`), including
   every model weight, and force the runtime offline
   (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`). This removed the single biggest
   cost sink: first-time setup on a paid machine.
2. **Stream every LLM request.** A non-streaming completion is silent while the
   model generates, and the host's port forward drops the idle socket after
   ~15 s. Measured **4/6 non-streaming vs 6/6 streaming**. This was the real
   cause of "every batch aborts with Remote end closed".
3. **Retry transient transport failures** in *both* clients (LLM and OCR). The
   OCR `/prepare` call is also non-streaming; the pilot survived 20 pages by
   luck, the full volume died at page 5.
4. **One machine, enforced.** `vast up` refuses to rent while one of ours is
   running/loading; `enforce_single()` + `vast_current.txt` record the one we
   own; `deploy/vast_watchdog.py` destroys any extra instance we own. A stray
   script once rented a second machine while the first was still billing.
5. **Vendor comic-text-detector at a pinned commit.** It cannot be
   pip-installed (no `setup.py`/`pyproject.toml`, no tags, not on PyPI) and its
   failure was hidden, so detection silently fell back to EasyOCR on every
   machine. Now the neural detector actually runs (`detector: true`).
6. **Diagnosability by design**: `faulthandler` to a log file, per-stage log
   markers, VRAM in `/health`, a `GET /logs` endpoint readable over HTTP after a
   crash, and a restart loop for the service.
7. **Host selection by pull speed**: Docker Hub throughput is per host. One host
   pulled the 15.5 GB image at 3.6 MB/s (16% in 12 min → ~60 min); a host with
   warm layers reached `running` in **187 s**. Prefer warm hosts; blacklist slow
   ones; watchdog the pull.
8. **Keep the machine when a run fails.** It is the debugging environment.
   Destroy only when the work is verified done.
9. **Verify by content, every time**: page count, page order, image dimensions,
   and translation coverage (blocks with source and non-empty translation).
10. **Audit detection quality objectively, and fix it adaptively.** Comparing the
    text-stroke density of each input page with its delivered page found **74 of
    1034 pages still carrying text that was never erased or translated** (~32 of
    them real text pages; one book had 20 consecutive pages missed). The server
    now runs the plain detector pass first and **escalates** to rotated
    (vertical text) and inverted (white-on-dark) passes only when that pass finds
    suspiciously little text, with `DET_SIZE` raised 1024 → 1280. The audit is
    free (no GPU, no viewing) and is the before/after measurement to use.

---

## 4. Verification that must pass before delivering

```
pages in == pages out            (never rename or drop a page)
page order identical
image dimensions unchanged
translation coverage: every block with source text has a non-empty translation
```

`tools/verify_batch.py --dir .` prints all of this. A non-zero exit means do not
deliver.

Measured example (all five volumes passed):

| Volume | Pages | Output | Blocks translated | Pages changed | Size mismatches |
|---|---|---|---|---|---|
| 1 | 237 | 107.9 MB | 5255 / 5255 | 237/237 | none |
| 2 | 257 | 192.8 MB | 4934 / 4934 | 257/257 | none |
| 3 | 16 | 37.1 MB | 232 / 232 | 16/16 | none |
| 4 | 250 | 188.7 MB | 3745 / 3745 | 250/250 | none |
| 5 | 237 | (last) | — | — | — |

---

## 5. Ten rules that keep the bill at ~15 cents per volume

1. **One machine at a time.** Never write a script that can rent.
2. **Never leave a machine idle.** If nothing is running, destroy it.
3. **Stream long requests and retry drops.**
4. **Prefer warm-cached hosts; blacklist slow ones.**
5. **Keep the machine on failure; destroy on verified success.**
6. **Report money first** in every report; the cap is **$1 per task**.
7. **Validate locally, for free, before renting** (mock OCR dry run, `bash -n`,
   `py_compile`, a build smoke test that blocks the network).
8. **Judge by content**, never by exit codes or log line counts.
9. **Identify work items unambiguously** — two different volumes had 237 pages.
10. **Never destroy a machine you still need for debugging.**

---

## 6. When it goes wrong, read this first

| Symptom | Most likely cause | Action |
|---|---|---|
| `Remote end closed connection` / `WinError 10054` | a non-streaming request was dropped by the port forward | ensure `stream: true` and retries; check `config.json` |
| the first OCR call fails on a baked image | the model wanted HuggingFace | `HF_HUB_OFFLINE=1`; verify the cache paths are baked |
| `detector: false` in `/health` | the vendored detector did not load | check `/logs`; confirm `/opt/comic-text-detector` exists |
| a volume dies at page ~5 but the pilot passed | OCR `/prepare` has no retry | retry it |
| two machines in the instance list | a leftover script rented | kill it, run the watchdog, keep one instance |
| the image pull never finishes | slow host | pull watchdog, switch to a warm host, blacklist it |
| `create instance` refused, no reason | args > 16384 chars, or a stale offer (`no_such_ask`) | gzip the payload; re-query offers |
| a log-based counter goes down | duplicated log lines | write the log file in one place only |
