# 20 — Hy-MT2 Colab Workflow (`hymt2-heretic-colab`)

> **Branch:** `hymt2-heretic-colab` (branched from `origin/hymt2-heretic` at `b10f481`)
> **Updated:** 2026-11-28 (+08:00), Milestone 12
> **Agent:** Claude Code (Milestone 12); Antigravity (Milestones 1-11)
> **Model:** Claude Sonnet 5.5 (Milestone 12); Gemini 4 Neon (Milestones 1-11)
> **Status:** Milestone 12 (guards ported from sakura Milestone 13) verified **offline only**: 16/16 tests and 3 hostile-model chaos runs on synthetic English input. **This branch has still never booted on a GPU**, and no real hymt2 output has been checked. Where sections 2.1-2.3 disagree with 2.4, **2.4 wins**.
> **Session Postmortem & Full Context:** See [21_colab_session_postmortem.md](21_colab_session_postmortem.md) for the complete C1–C12 failure table, 10-item operator accountability log, F1–F11 review fixes, 9 permanent guardrails, and live Colab L4 specs.

---

## 1. Identity & Scope

| Field | Value |
|---|---|
| Agent harness | Antigravity |
| Model | Gemini 4 Neon |
| Branch | `hymt2-heretic-colab` |
| Base commit | `b10f481` (`origin/hymt2-heretic`) |
| Target directory | `deploy/colab-hymt2/` |
| Touched existing files | `llm_handoff/INDEX.md` only (Dockerfiles and `.github/workflows/**` untouched) |

### Purpose
Provide a zero-compile **Google Colab L4 (24 GB) + Google Drive (5 TB) one-click workflow** for `Hy-MT2-30B-A3B-uncensored-heretic` (English -> Taiwan Traditional Chinese adult galgame H-scene translation), bypassing Vast.ai rental and `hanhua.py vast up` limitations while preserving all model/prompt decisions from `deploy/hymt2-heretic/`:
1. **Free CPU GGUF & Binary Deployer (`deploy_models_colab.ipynb`):** Runs on a free CPU runtime, mounts Drive, creates fixed folders, downloads pinned GGUF weights from HuggingFace (with `hf-mirror` fallback) plus the prebuilt `llama-server` tarball (`colab-l4-llama-b12019`), and verifies exact size and SHA-256 on both local SSD and Drive before atomic rename.
2. **Colab L4 One-Click Translator (`hymt2_colab.ipynb` + `runner.py`):** Extracts the pinned prebuilt `llama-server-l4-b12019.tar.gz` bundle from Drive `bin/` (or GitHub Release fallback) with zero compilation, stages the 18.24 GB GGUF once to local NVMe SSD, enforces 4 hard GPU gates (`>= 22,000 MiB` total VRAM, `CUDA0`, full layer offload at `-lv 4`, `>= 18,000 MiB` used VRAM, 3-line plain/explicit/SFX Gate `>= 8 tok/s`), runs 1-line-per-request 4-way parallel translation with `STYLE_ADULT` personalization prompt, `tw2sp` glossary/speaker, empty history for `ui`/`system`, plain-template mechanical fallback, OpenCC `t2s` canonical + `s2twp` default output, watch-char QA, writes `.DONE.json`, flushes Drive, and protects GPU quota via `DISCONNECT_ON_ERROR=False` + `IDLE_UNASSIGN_MINUTES=15`.
3. **Local Option A Drive Bridge (`drive_bridge.py` + `skill/SKILL.md`):** Pure-stdlib Windows CLI that checks `GoogleDriveFS.exe`, auto-detects the single mounted Google Drive containing `galgame_mtl` (supporting multi-account Drive e.g. `H:\`), validates strict `.jsonl` + `glossary.json`, syncs `schema.py`/`runner.py` to Drive `scripts/hymt2/`, atomically pushes `<job>.jsonl` + `<job>.manifest.json`, checks status, and pulls verified outputs after `.DONE.json` SHA-256 verification.
4. **Agent Migration SOP (`AGENT_SOP_COLAB.md`):** Explicit migration guide for agents arriving with `vast.ai` handoff context plus EN->ZH adult H-scene specifics.

---

## 2. Locked Constants, Quality Contracts & VRAM Note

### 2.1 Runtime & Model Constants
| Item | Value |
|---|---|
| Colab Drive root | `/content/drive/MyDrive/galgame_mtl` |
| Model | `mradermacher/Hy-MT2-30B-A3B-uncensored-heretic-i1-GGUF` (`Hy-MT2-30B-A3B-uncensored-heretic.i1-Q4_K_M.gguf`) |
| Pinned HF rev | `862b7d66847b339c0594b8de39d38b035ee75f25` |
| Exact model size & SHA-256 | `18,236,703,488` bytes / `b0358d0319aae3f821fef4989bd85f409336139bfc7db59fd049848b9882321c` |
| Prebuilt server bundle | GitHub Release `poynt2005/hanhua-mtl` tag `colab-l4-llama-b12019`, `llama-server-l4-b12019.tar.gz`, `773,245,209` bytes, SHA-256 `802a169e33dde5e73403c2a36f3d6dbfd2d34d5e9ccd898e79a1eb899bcac947`, `43` entries |
| Server flags | `127.0.0.1:11434`, ctx `32768`, `-np 4` (`8192`/slot), `-ngl 999`, `--jinja`, `--device CUDA0 --cont-batching -lv 4`, `LLAMA_ARG_UI=0` (never pass redundant `LLAMA_ARG_LOG_VERBOSITY`) |
| GPU Gates | Gate 0: total VRAM >= `22,000 MiB`; Gate 1: `CUDA0`; Gate 2: `-lv 4` `offloaded X/X layers to GPU` (polled up to `10 x 0.5s`); Gate 3: used VRAM >= `18,000 MiB`; Gate 4: 3-line Gate >= `8.0 tok/s` |
| Sampling | `temperature=0.7`, `top_p=1.0`, `top_k=0`, `repeat_penalty=1.0`, `max_tokens=1024`, `stream=True` |
| Traditional conversion | Default OpenCC `s2twp` (per `19_galgame_image_branches.md`; `s2t` selectable via `--tw-mode`) |

### 2.2 Refusal Detection Split (C11)
- **Gate 4 (`run_preflight_gate`)**: Runs both `mechanical_problems()` and broad `_GATE_REFUSAL_RE` (`抱歉|無法|无法|不能提供|I can't|I cannot|I'm sorry|as an AI`), matching `origin/hymt2-heretic:deploy/hymt2-heretic/hymt2_prompt.py`, plus plain-template fallback (F3).
- **Normal Dialogue (`mechanical_problems`)**: Uses strict AI-boilerplate `_REFUSAL_RE` (`as an AI`, `I cannot fulfill/assist/translate this`, `I'm sorry, I can't/cannot`, `不能提供相关/色情/成人`, policy citation patterns) so ordinary translated dialogue such as `"抱歉，我无法停下来。"` is never falsely flagged as a refusal.

### 2.3 Option A Exit Codes, FUSE Resilience, Notebook Lifecycle & First-Boot L4 VRAM Note
- **Three-tier exit codes (`runner.py` + `hymt2_colab.ipynb`):**
  - `EXIT_OK = 0`: All jobs passed 100%, `.DONE.json` written -> `[DONE]` -> `AUTO_DISCONNECT=True` unassigns L4.
  - `EXIT_PARTIAL = 2`: All units completed while `llama-server` stayed healthy (`proc.poll() is None`), outputs + `checkpoint.jsonl` + `failures.jsonl` saved to Drive, `.DONE.json` NOT written -> notebook prints calm `[PARTIAL]` summary (no `RuntimeError`), flushes Drive, and disconnects under `AUTO_DISCONNECT=True` (no 15-min idle burn).
  - `EXIT_FATAL = 1`: Startup Gates 0-4, invalid inputs/manifest, persistent checkpoint FUSE error, or mid-run `llama-server` exit (`proc.poll() is not None`) -> raises/returns fatal error, `DISCONNECT_ON_ERROR=False` prints `STOP THE RUNTIME NOW` and retains L4 up to `IDLE_UNASSIGN_MINUTES=15`.
- **Drive FUSE resilience:** `_append_checkpoint_rows` retries transient `OSError` up to `5` times (`0.25s` exponential backoff) inside `_flush_pending` before updating `done_map` / `progress_counter`; `_log_gpu_progress` isolates `gpu.log` `OSError` as a non-fatal warning.
- **L4 VRAM headroom warning (`not tested` live):** Observed Colab L4 free VRAM before model load was `22,369 MiB` during Sakura Run #4, and `llama.cpp` `b12019` defaults `-fit on` (targeting `>= 1,024 MiB` free device memory). At `CTX_SIZE = 32768`, `Hy-MT2-30B-A3B` weights (~17.0 GiB) + KV cache (~3.0 GiB) + compute buffers project to `~20.8-21.2 GiB` — very close to the `-fit on` threshold. On the first live L4 boot, inspect `hymt2/work/llama.log` (`common_params_fit_impl` lines); if `-fit on` spills layers to CPU and fails Gate 2, lowering `CTX_SIZE` from `32768` to `16384` (4,096/slot, still ample for 1-line requests) frees ~1.5 GiB immediately.

### 2.4 Milestone 12: broken-output guards (supersedes older text above)

Ported by hand from `sakura-ollama-colab` Milestone 13, where a real job delivered runaway repetition and damaged control codes that had passed QA. **No such defect has been observed on hymt2** (it has no live run); the same gaps existed in its code (no `finish_reason` check, no length/repetition check, fixed `max_tokens`).

- `finish_reason=length` -> `OutputTruncated`, never accepted. `max_tokens` = min(1024, max(96, 2 x chars + 48)). Sampling is unchanged.
- Inline control codes `_name!n,n,` are split out and re-inserted by the runner. **Each segment is its own request** (earlier source segments are passed as context). Newlines are **not** split (unchanged behaviour). `control_code_mismatch` is validated; codes are ignored by the "has English" and "not translated" checks.
- Runaway: `too_long` (> 2x source + 30), `repetition` (single-char run > max(24, 2 x source letter run + 4), or a Chinese phrase looping 8+ times when the English source has no loop of its own). Letter runs > 6 are shortened in the prompt only. **These thresholds are untested against real hymt2 output.**
- Per segment, 3 tries: personalization template -> card plain template -> personalization without context. Then the **whole unit keeps the English source**. No interjection table and no Japanese kana/quote rules were ported.
- Stable SHA-1 `unit_key` checkpoint (legacy `wid` rows migrate using the old numbering rule); cached rows are revalidated on load and at assembly.
- `[DONE]` when kept-source rows <= floor(total x 0.001) (`--max-untranslated-ratio`), `missing = 0`, and no unit failed on transport. `qa.failures` = kept units and may be non-zero in a DONE job.
- `work/<job>.attempts.jsonl` (rejected raw replies, append-only) and `output/<job>.review.jsonl` (covered by DONE, copied by `pull`; fill `translation` and save as `input/<job>.overrides.jsonl`).
- New CLI: `--repair-only`, `--job`, `--overrides`, `--assemble-only`. Notebook adds `REPAIR_ONLY`, `TARGET_JOB`.
- Verification: 16/16 tests (one old test changed on purpose: a failed line is now kept source, not `missing`); 3 chaos seeds on 8,000 synthetic rows with ~4k injected faults each: no invariant violation.

---

## 3. Progress Log

| Timestamp (+08:00) | Milestone | Outcome | Notes |
|---|---|---|---|
| 2026-11-26 01:30 | Milestone 1: Branch & Core Implementation | Done | Created `deploy/colab-hymt2/` (`schema.py`, `runner.py`, `drive_bridge.py`, `deploy_models_colab.ipynb`, `hymt2_colab.ipynb`, `AGENT_SOP_COLAB.md`, `skill/SKILL.md`, `README.md`, `test_all.py`) and updated `llm_handoff/INDEX.md` |
| 2026-11-26 01:32 | Milestone 2: Code Review, Offline Tests & Sanitization | Done | Verified `b12019` `LLAMA_ARG_*` env vars, `tw2sp`/`s2twp` one-to-many watch-char behavior (`干/發/後`), `llama.log` fd closure, `try/except` around `drive.flush_and_unmount()`, and Drive glossary presence warning. `py_compile` + `test_all.py` (5/5 tests) passed; sanitization scan 0 hits |
| 2026-11-26 11:52 | Milestone 3: Colab Logging, CMake & Drive FUSE Parity Fixes | Done | Ported live `Popen` streaming + `hymt2/work/runner.log` tee, early Gate 0, single-copy NVMe `stage_model`, `copyfile`, and OS metadata ignore |
| 2026-11-26 12:02 | Milestone 4: Disable `llama-ui` Prebuilt Fetch for `b12019` | Done | Ported UI asset build disable flags before eliminating compilation entirely in Milestone 5 |
| 2026-11-26 12:33 | Milestone 5: Zero-Compile Prebuilt Bundle & Full Build Removal | Done | Switched exclusively to GitHub Release `colab-l4-llama-b12019` (`llama-server-l4-b12019.tar.gz`, `773,245,209` bytes, SHA-256 `802a169e...`) cached in Drive `bin/`, removing all build code |
| 2026-11-26 13:21 | Milestone 6: Fix `b12019` Log Verbosity (`-lv 4`) for Gate 2 GPU Offload Verification | Done | Ported `-lv 4`, `proc.terminate()`, and `llama.log` tail error reporting |
| 2026-11-26 13:35 | Milestone 7: Complete Code Review Fixes (F1, F3, F4, F5, F6, F7) | Done | Polled `llama.log` up to `10 x 0.5s` (F1); added plain-template preflight fallback (F3); added `DISCONNECT_ON_ERROR` toggle (F4); switched `drive_bridge.py` to `copyfile` + `Prebuilt bin` check (F5); batched checkpoint flush every 10 units + `finally` flush (F6); moved GPU logging under `gpu_log_lock` (F7) |
| 2026-11-26 14:32 | Milestone 8: Parity Sync from Sakura C9 (`DISCONNECT_ON_ERROR = False` & Clean `-lv 4`) | Done | Removed redundant `LLAMA_ARG_LOG_VERBOSITY=4` env var (keeping `-lv 4`), defaulted `DISCONNECT_ON_ERROR = False`, updated `test_all.py` assertions |
| 2026-11-28 00:52 | Milestone 9: Port `IDLE_UNASSIGN_MINUTES=15` Safeguard & Sync Postmortem C10 | Done | Added `IDLE_UNASSIGN_MINUTES = 15` bounded countdown and `STOP THE RUNTIME NOW` banner after Drive flush in `hymt2_colab.ipynb`; synced `21_colab_session_postmortem.md` |
| 2026-11-28 01:02 | Milestone 10: Narrow `_REFUSAL_RE` in `mechanical_problems()` to Prevent Dialogue False Positives (C11) | Done | Separated `_GATE_REFUSAL_RE` (Gate 4 only) from strict AI-boilerplate `_REFUSAL_RE` in `mechanical_problems()`, preventing normal English->Chinese dialogue (`"抱歉，我无法……"`) from being falsely flagged as refusals; synced `runner.py` to Drive `scripts/hymt2/` |
| 2026-11-28 01:32 | Milestone 11: Option A Exit-Code Separation (`0/1/2`) & FUSE I/O Resilience (C12) | Done | Separated `EXIT_OK = 0`, `EXIT_FATAL = 1`, `EXIT_PARTIAL = 2` so completed runs with minor failures save all outputs, skip `.DONE.json`, print calm `[PARTIAL]` summary in `hymt2_colab.ipynb` (no `RuntimeError`), and still honor `AUTO_DISCONNECT=True`; added `proc.poll() is not None` mid-run server crash guard, 5-attempt exponential-backoff `_append_checkpoint_rows` (committing `done_map` post-write), and `gpu.log` `OSError` warning isolation; added 4 tests (`10/10` pass) |
| 2026-11-28 | Milestone 12: broken-output guards, control-code segmentation, `unit_key`, overrides/repair/assemble, review + attempts files, 0.1% DONE rule | Done offline | See 2.4. 16/16 tests + chaos runs. Never run on a GPU |

---

## 4. Verification Status (Honest State)

| Check | State | Notes |
|---|---|---|
| `py_compile` on `schema.py`, `runner.py`, `drive_bridge.py`, `test_all.py` | `done` (pass) | Executed after Milestone 11 edit |
| Offline `test_all.py` (10 tests: schema, `tw2sp`/`s2twp`, whole-word glossary, personalization + plain fallback, strict refusal vs normal dialogue non-false-positive, `ui` empty-style/context, deduplication, resume, watch-chars, zero-compile prebuilt bundle + SHA-256 verification + zero `cmake`/`--rebuild-bin` assertions + `-lv 4` assertion, OS metadata ignore, `drive_bridge.py`, `.ipynb` JSON + `ast.parse` + `runner.log` + `[PARTIAL]` + disconnect matrix, `EXIT_PARTIAL=2`, Drive FUSE retry & `gpu.log` resilience, mid-run server crash `EXIT_FATAL`) | `done` (10/10 pass) | Executed after Milestone 11 edit using local mock SSE & HTTP servers and temp dirs |
| Code review, sanitization audit & Drive `scripts/hymt2/runner.py` SHA-256 sync | `done` (0 hits) | Verified zero secrets, zero local usernames/paths, zero copyrighted game strings, zero Docker CI triggers |
| Live Google Colab L4 GPU run (`hy_v3` load, `-fit on` VRAM margin, English refusal rate, H-scene quality, `s2twp` spot check) | `not tested` | Requires operator live Colab L4 run (`deploy_models_colab.ipynb` with `DEPLOY_HYMT2=True` then `hymt2_colab.ipynb`) |

---

## 5. First Live L4 Boot Checklist & Post-Pull Cleanup
1. Ensure 18.24 GB `Hy-MT2-30B-A3B-uncensored-heretic.i1-Q4_K_M.gguf` and `llama-server-l4-b12019.tar.gz` are deployed via `deploy_models_colab.ipynb` on Free CPU Colab.
2. Push a small ~50-line English adult-scene spot-check job with `drive_bridge.py push` (or `python <instructions_dir>/drive_bridge.py push` when working outside the repo).
3. Open latest `hymt2_colab.ipynb` on **Colab L4 GPU** (verify `DISCONNECT_ON_ERROR=False`, `IDLE_UNASSIGN_MINUTES=15`, `AUTO_DISCONNECT=True`) and run `Ctrl+F9`.
4. After completion (`[DONE]` `rc=0` or `[PARTIAL]` `rc=2`), inspect `hymt2/work/llama.log` (`common_params_fit_impl` + `offloaded` + tok/s), `hymt2/output/<job>.qa_report.json` (and `hymt2/work/<job>.failures.jsonl` if partial), pull verified outputs with `drive_bridge.py pull`, and manually spot-check explicit tone and `s2twp` `干`/`發`/`後` conversions before running full game scripts.
5. **Post-pull Drive cleanup:** ONLY after `pull` passes SHA-256 + ID parity and local outputs are verified saved on disk, delete the completed job's residual files from `hymt2/input/`, `hymt2/work/`, and `hymt2/output/` so the next Colab job never scans stale files. **Never** delete `models/`, `bin/`, or `scripts/`.

### `rc = 2` / `rc = 0` Recovery & Retest Playbook
1. **Review:** read `hymt2/output/<job>.review.jsonl` (raw rejected replies: `hymt2/work/<job>.attempts.jsonl`).
2. **0-GPU rebuild:** fill `translation`, save as `hymt2/input/<job>.overrides.jsonl`, then run
   `python <instructions_dir>/runner.py --drive-root "<drive_root>" --assemble-only --job <job> --manifest-wait 0`
3. **Model retry:** adjust `glossary.json` / partial overrides, set `REPAIR_ONLY = True` and `TARGET_JOB`, rerun the latest notebook. Passing rows are reused.
4. **Force retranslation:** delete the rows (by `unit_key`) or the whole checkpoint, rerun with `REPAIR_ONLY = True` or `FORCE_RERUN = True`.

> **Standalone instructions export:** Every export to `<instructions_dir>/hymt2-heretic-colab/` must bundle all **11 files** (the 9 files under `deploy/colab-hymt2/` plus `20_hymt2_colab_workflow.md` and `21_colab_session_postmortem.md`).
