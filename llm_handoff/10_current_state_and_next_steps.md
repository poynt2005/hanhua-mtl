# 10 — Current state and next steps

## Exactly where things stand

| Item | State |
|---|---|
| Core pipeline (`unpack→extract→translate→render→repack`) | **works**, verified on a 5-volume batch |
| `auto` one-shot command | works; default output `<stem>_MTL_zhTW<ext>` |
| OCR/inpaint service (`deploy/ocr_server.py`) | works; neural detector mask, LaMa erase, crops back to the original size, per-stage logging, `GET /logs`, restart loop |
| Local CPU path | works; inpainting limited to lightweight methods (no LaMa without torch) |
| GPU image (`deploy/Dockerfile`) | **v0.2.3 pushed** (~15.5 GB): every model weight baked into fixed cache paths, runtime forced offline, comic-text-detector vendored at a pinned commit, build smoke test that blocks all outbound sockets |
| vast.ai controller (`mth/vast.py`) | **single-instance invariant** (`up` refuses while one of ours is running/loading), recorded owner (`vast_current.txt`), `enforce_single()`, refusal reasons surfaced |
| Watchdog | `deploy/vast_watchdog.py` destroys any extra instance we own (never the user's) |
| LLM client | **streams** responses (SSE) and retries transient drops - this was the root cause of every aborted batch |
| OCR client | retries `/prepare` and names the failing endpoint |
| Cost-safety docs | `deploy/VASTAI_PLAYBOOK.md` (rules + pitfall table), `13_operating_manual.md` (the runbook), `skills/vastai-cost-safety/` |
| Term consistency | implemented (extract / review / translate / normalize / verify) |
| Deliverables | produced; **gitignored** (never committed) |

## Verified batch run (the one that proves the whole flow)

One rented machine (Tesla V100 32 GB @ $0.153/hr), five volumes, sequential,
each verified before the next started:

| # | Volume | Pages | Output | Translation coverage | Order / size |
|---|---|---|---|---|---|
| 1 | `[BJ01619195] …` | 237 | 107.9 MB | 5255 / 5255 blocks | same order, no size change |
| 2 | `[Kabashima Akira] Tokushu Play …` | 257 | 192.8 MB | 4934 / 4934 blocks | same order, no size change |
| 3 | `RJ134098.zip` | 16 | 37.1 MB | 232 / 232 blocks | same order, no size change |
| 4 | `[Kabashima Akira] Jinzou Manko …` | 250 | ~189 MB | 3745 / 3745 blocks | same order, no size change |
| 5 | `[BJ02441076] …` | 237 | (final volume) | — | — |

Measured throughput: **~15 s/page** (OCR ~6.9 s + translation ~8.3 s), i.e. a
240-page volume is ~60 minutes and **~$0.15**. The whole 5-volume batch stayed
**under $1**. Details and the cost model: `13_operating_manual.md`.

Each volume was verified for: page count, page order, image dimensions, and
translation coverage (every block with source text has a non-empty translation).

## What actually fixed the batch failures (in order of impact)

1. **Stream the LLM request.** Non-streaming completions are silent while the
   model generates; the host's port forward drops the idle socket after ~15 s
   (measured 4/6 non-streaming vs **6/6 streaming**). See G11.
2. **Retry the OCR `/prepare`.** It is also non-streaming; the pilot survived 20
   pages by luck, the full volume died at page 5. See G11.
3. **Vendor comic-text-detector** at a pinned commit - it can never be
   pip-installed, and its failure was hidden, so detection silently fell back to
   EasyOCR on every machine. See G10.
4. **Force the runtime offline** and bake every weight into fixed cache paths -
   `transformers` phones HuggingFace on every load even with a warm cache. See G9.
5. **One machine, enforced** - a leftover script rented a second machine while
   the first was still billing. See C8.
6. **Diagnosability by design** - `faulthandler`, stage markers, VRAM in
   `/health`, `GET /logs`, restart loop. See G12.
7. **Host selection by pull speed** - Docker Hub throughput is per host; a warm
   host booted the 15.5 GB image in 187 s, a slow one needed ~60 min. See C12.
8. **Keep the machine on failure** - it is the debugging environment. See C7.

## Known issues / open items

1. **Erase and translation quality have not been reviewed by the user.** The
   pipeline output is machine translation; a human pass has not happened.
2. **The `openai` profile** (`_translate_openai`, JSON mode) is now streaming
   too, but only the `sakura` profile has been exercised at volume.
3. **`terms`/`normalize`** use the translation model for JSON work, which it is
   poor at; a general instruction model would be better.

## Prioritised next steps

### P0 — quality (only if the user asks)
- Have the user review one finished volume: erase quality and translation.
- Optional: rebuild the image with **Q6_K** instead of Q4_K_M. Measured cost:
  **+$0.036 per 250-page volume**, same 24 GB GPU class (no need for a bigger
  card). bf16 would need a 40-80 GB card at 4-6x the price - not worth it.

### P1 — hardening
- Pin the base image by digest.
- Add a `--strict` end-to-end gate that refuses to deliver unless `verify` passes
  (page count, order, dimensions, coverage).

### P2 — nice to have
- Optional PSD export for manual touch-up.
- Better vertical-text fitting; wrap Latin text at word boundaries.

## Rules to keep following

- Never commit archives, pipeline artifacts, `config.json`, keys, or any input
  content (see `.gitignore`).
- Money first in every report; **$1 cap per task**.
- **One machine at a time**; keep it while it is useful; destroy when done.
- Never write ad-hoc rental code (see `deploy/VASTAI_PLAYBOOK.md` R7).

## If you are picking this up

1. Read `INDEX.md`, then `13_operating_manual.md`, then `01`→`12`.
2. Run `selftest` and `auto --ocr mock` to confirm the pipeline locally (free).
3. Rent one machine with `hanhua.py vast up --baked`, run the three gates, then
   the batch through `_chain_run.py` (which verifies each volume and tears down).
4. Ask the user to review the output before spending more on machines.
