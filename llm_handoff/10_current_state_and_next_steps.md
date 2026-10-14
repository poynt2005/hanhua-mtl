# 10 — Current state and next steps

## Exactly where things stand

| Item | State |
|---|---|
| Core pipeline (`unpack→extract→translate→render→repack`) | **works**, verified on a 5-volume batch |
| `auto` one-shot command | works; default output `<stem>_MTL_zhTW<ext>` |
| OCR/inpaint service (`deploy/ocr_server.py`) | works; neural detector mask, LaMa erase, crops back to the original size, per-stage logging, `GET /logs`, restart loop |
| Local CPU path | works; inpainting limited to lightweight methods (no LaMa without torch) |
| GPU image (`deploy/Dockerfile`) | **v0.5.0** (~18.7 GB): every model weight baked into fixed cache paths, runtime forced offline, comic-text-detector vendored at a pinned commit, **Sakura-14B-Qwen3-v1.5 at Q6_K** (imported from its GGUF via a Modelfile, 12.12 GB), **16384-token context**, **adaptive multi-variant detection**, build smoke test that blocks all outbound sockets |
| Translation model | `sakura-qwen3-q6k`: the newest 14B Sakura (Qwen3 base, 2026-01) at Q6_K, with **that release's own recommended system prompt** (the Qwen3 line is tuned with different wording from the older GalTransl one) |
| Detection | plain pass first; **escalates** to rotated (vertical text) and inverted (white-on-dark) passes only when it finds < 5 boxes; `DET_SIZE` 1280; rotated boxes mapped back and deduped by IoU. All knobs env-overridable |
| Pre-flight gate | `_gate.py`: translates 3 lines and requires exactly 3 non-empty Chinese lines with **no reasoning/template noise**, plus OCR detector health. Run it before every batch |
| Unit tests (free, no torch) | `deploy/test_detection.py`: rotation mappings against numpy ground truth, dedup, and the adaptive escalation flow with a fake detector |
| Quality audit (free, no GPU) | compare each **input** page with its **delivered** page by text-stroke density; this is what produced the 74-page finding below |
| vast.ai controller (`mth/vast.py`) | **single-instance invariant** (`up` refuses while one of ours is running/loading), recorded owner (`vast_current.txt`), `enforce_single()`, refusal reasons surfaced |
| Watchdog | `deploy/vast_watchdog.py` destroys any extra instance we own (never the user's) |
| LLM client | **streams** responses (SSE) and retries transient drops - this was the root cause of every aborted batch |
| OCR client | retries `/prepare` and names the failing endpoint |
| Cost-safety docs | `deploy/VASTAI_PLAYBOOK.md` (rules + pitfall table), `13_operating_manual.md` (the runbook), `skills/vastai-cost-safety/` |
| Term consistency | implemented (extract / review / translate / normalize / verify) |
| Deliverables | **6 volumes, 1034 pages** delivered to `ComicsZHTW/` (647 MB) with `MANIFEST.txt` and per-volume `.project.json`; all verified (page count, order, dimensions, 100% coverage). Archives are **gitignored** (never committed) |
| Repo tools | `tools/`: `gate.py`, `pull_watch.py`, `verify_batch.py`, `run_batch.py`, `audit_detection.py`, `audit_sheets.py`, `collect.py` (+ `deploy/test_detection.py` unit tests) |
| Machines | **0 running**; the whole session cost **$0.974** |

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

**Model versions in the delivery** (important): the batch above used the older
`GalTransl-14B-v3.8` at Q4_K_M with an 8192 context. Only the 250-page Jinzou
volume was later **re-run on `v0.5.0`** (Sakura-14B-**Qwen3**-v1.5 at Q6_K, 16384
context, its own recommended prompt): 3794/3794 blocks, verified, 12.9 s/page.
The other five volumes still carry the older model's output; re-running them on
v0.5.0 costs ~$0.60 (see `15_run_log.md` §9).

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
9. **Audit detection objectively and fix it adaptively.** Comparing every input
   page with its delivered page found **74 of 1034 pages still carrying text that
   was never erased or translated** (~32 of them real text pages; one volume had
   20 consecutive pages missed). The server now escalates to rotated/inverted
   detector passes when the plain pass finds suspiciously little text, at
   `DET_SIZE` 1280. See G16 and `13_operating_manual.md` item 10.

## Detection quality: what the audit found, and the correction

All delivered pages passed the usual checks (page count, order, dimensions,
100% of *detected* blocks translated) - but those say nothing about text that was
never detected, so a free audit was written: compare the **text-stroke density**
of each input page with its delivered page (no GPU, no viewing, no money).

First result: **74 of 1034 pages still looked text-like and unchanged**, ~32 with
thousands of strokes, and one volume had 20 consecutive such pages. It looked like
a major defect, so an adaptive multi-variant detector was built and shipped
(`v0.5.0`), and the 250-page volume was re-run to measure it:

| Metric | Before (single pass) | After (adaptive) |
|---|---|---|
| blocks | 3745 | 3794 (+49) |
| pages with <=2 blocks | 30 | 29 |
| of those, still text-like | 23 | 23 |

**The fix did not move the metric - and it is not claimed as a fix.**

Then the user looked at the contact sheets: **the flagged pages are almost
entirely onomatopoeia / sound effects**, and this pipeline sets
`keep_sfx: true`, i.e. keeping SFX is intended.

So the audit **over-called**: the stroke metric cannot tell dialogue from SFX.
`tools/audit_detection.py` now says so in its docstring and output, and
`14_thrashing_postmortem.md` / `15_run_log.md` record it as a wasted chase. The
adaptive detection was kept (harmless, +49 blocks) but the honest position is
that **detection was not broken**.

**Lesson: an automated quality metric is a triage list, not a defect count. Build
the contact sheets and have a human look before changing code.**

## Known issues / open items

1. **Erase and translation quality have not been reviewed by the user.** The
   pipeline output is machine translation; a human pass has not happened.
2. **The detection fix has not been re-measured on a full volume yet.** The
   v0.5.0 pilot is running the same 250-page volume so the before/after is a
   number (see above).
3. **The delivered batch used the OLD model and the OLD detection.** The five
   volumes already produced are v0.2.2/v0.4.0-era output: Q4_K_M, 8192 context,
   single-pass detection. Re-running them with v0.5.0 is what improves them.
4. **The `openai` profile** (`_translate_openai`, JSON mode) is now streaming
   too, but only the `sakura` profile has been exercised at volume.
5. **`terms`/`normalize`** use the translation model for JSON work, which it is
   poor at; a general instruction model would be better.

## Prioritised next steps

### P0 — finish the quality pass
- Wait for the v0.5.0 pilot result and compare the audit numbers with the
  "before" (30 low-block pages, 23 still carrying text).
- Then re-run the volumes whose audit is worst (250-page and 257-page volumes
  had the most low-block pages). Note the new per-volume cost: Q6_K is ~25%
  slower and escalation adds detector passes, so budget ~$0.25-0.30/volume.
- Have the user review one finished volume before re-running everything.

### P1 — hardening
- Pin the base image by digest.
- Add a `--strict` end-to-end gate that refuses to deliver unless `verify` passes
  (page count, order, dimensions, coverage).
- Move the useful helper scripts into `tools/` (parameterised): `_gate.py`,
  `_verify_all.py`, `_audit_compare.py`, `_ocr_audit.py`, `_chain_run.py`.

### P2 — nice to have
- Consider `PaddleOCR-VL-For-Manga` (Apache-2.0, 70% full-sentence accuracy on
  Manga109-s crops) as a **fallback** OCR for the pages the detector still
  struggles with - not a full replacement (a VLM OCR is 3-6x slower).
- Optional PSD export for manual touch-up.
- Better vertical-text fitting; wrap Latin text at word boundaries.

## Rules to keep following

- Never commit archives, pipeline artifacts, `config.json`, keys, or any input
  content (see `.gitignore`).
- Money first in every report; **$1 cap per task**.
- **One machine at a time**; keep it while it is useful; destroy when done.
- Never write ad-hoc rental code (see `deploy/VASTAI_PLAYBOOK.md` R7).
- **Never rent for an image tag that is not published yet** (V64) - check the
  registry first.
- Run `_gate.py` before any batch; audit with `_audit_compare.py` before claiming
  a quality improvement.

## If you are picking this up

1. Read `INDEX.md`, then `13_operating_manual.md`, then
   `14_thrashing_postmortem.md` (it will save you money), then `01`→`12`.
2. Run `selftest` and `auto --ocr mock` locally (free), and
   `python deploy/test_detection.py`.
3. Confirm the image tag exists on the registry, rent one machine with
   `hanhua.py vast up --baked`, run `_gate.py`, then the batch through
   `_chain_run.py` (it verifies each volume and tears down on success).
4. Audit the delivered pages with `_audit_compare.py`; do not claim quality from
   impressions.
5. Ask the user to review the output before spending more on machines.
