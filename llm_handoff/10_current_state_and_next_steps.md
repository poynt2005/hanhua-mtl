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
9. **Audit detection objectively and fix it adaptively.** Comparing every input
   page with its delivered page found **74 of 1034 pages still carrying text that
   was never erased or translated** (~32 of them real text pages; one volume had
   20 consecutive pages missed). The server now escalates to rotated/inverted
   detector passes when the plain pass finds suspiciously little text, at
   `DET_SIZE` 1280. See G16 and `13_operating_manual.md` item 10.

## Detection quality: the measured gap and the fix

The first delivered batch looked fine by the usual checks (page count, order,
dimensions, 100% of *detected* blocks translated) but was **not** complete: a
single detector pass had silently missed whole regions, so those pages kept
their original Japanese.

How it was found (free - no GPU, no viewing, no money): for every page, measure
the **text-stroke density** (Otsu dark ratio + count of text-stroke-sized
connected components) of the *input* page and of the *delivered* page. If the
delivered page still has the same density, nothing was erased there.

| Measure | Result |
|---|---|
| Pages with ≤2 detected blocks (all 6 volumes) | 87 |
| Of those, delivered page unchanged AND still text-like | **74** |
| Of those, with >2000 strokes (near-certain text pages) | **~32** |
| Worst case | `Tokushu` pages **201-220** - 20 consecutive pages, all missed |
| Pages that were genuinely blank/credits (my earlier over-count) | ~13 |

Fix (v0.5.0): the plain pass runs first; if it finds fewer than
`DET_ESCALATE_BELOW` (5) boxes, the server also runs **rot90 / rot270** (vertical
lines become horizontal - the same idea as manga-image-translator's
`det_auto_rotate`) and **invert** (white-on-dark) passes, maps the boxes back to
the original coordinates (unit-tested) and dedups by IoU. Cost is low because
most pages never escalate.

**This is not yet re-measured on a full volume.** The v0.5.0 pilot translates the
same 250-page volume whose "before" numbers are known (30 low-block pages, 23 of
them still carrying text) and re-runs the same audit, so the improvement is a
number, not an impression.

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
