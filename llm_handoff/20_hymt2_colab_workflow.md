# 20 — Hy-MT2 Colab Workflow (`hymt2-heretic-colab`)

> **Branch:** `hymt2-heretic-colab` (branched from `origin/hymt2-heretic` at `b10f481`)
> **Updated:** 2026-11-28 01:06 +08:00
> **Agent:** Antigravity
> **Model:** Gemini 4 Neon
> **Status:** READY — Milestone 10 complete (zero-compile prebuilt bundle, `DISCONNECT_ON_ERROR=False` + `IDLE_UNASSIGN_MINUTES=15` safeguard, Gate 4 `-lv 4` + 5s async flush poll, plain-template preflight fallback, batched checkpoints, and C11 separated `_GATE_REFUSAL_RE` vs strict AI-boilerplate `_REFUSAL_RE`; 6/6 offline tests & sanitization passed; live L4 GPU boot still `not tested`)
> **Session Postmortem & Full Context:** See [21_colab_session_postmortem.md](21_colab_session_postmortem.md) for the complete C1–C11 failure table, 9-item operator accountability log, F1–F7 review fixes, 8 permanent guardrails, and live Colab L4 specs.

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

### 2.3 Notebook Lifecycle & First-Boot L4 VRAM Note
- **Notebook defaults:** `AUTO_DISCONNECT=True`, `DISCONNECT_ON_ERROR=False`, `IDLE_UNASSIGN_MINUTES=15`, `TW_CONVERT_MODE="s2twp"`. On error with `DISCONNECT_ON_ERROR=False`, Drive flushes first, prints `STOP THE RUNTIME NOW` + idle cost warning (~2 compute units/hr), and waits `IDLE_UNASSIGN_MINUTES` (interruptible via KeyboardInterrupt) before auto-unassigning.
- **L4 VRAM headroom warning (`not tested` live):** Observed Colab L4 free VRAM before model load was `22,369 MiB` during Sakura Run #4, and `llama.cpp` `b12019` defaults `-fit on` (targeting `>= 1,024 MiB` free device memory). At `CTX_SIZE = 32768`, `Hy-MT2-30B-A3B` weights (~17.0 GiB) + KV cache (~3.0 GiB) + compute buffers project to `~20.8-21.2 GiB` — very close to the `-fit on` threshold. On the first live L4 boot, inspect `hymt2/work/llama.log` (`common_params_fit_impl` lines); if `-fit on` spills layers to CPU and fails Gate 2, lowering `CTX_SIZE` from `32768` to `16384` (4,096/slot, still ample for 1-line requests) frees ~1.5 GiB immediately.

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

---

## 4. Verification Status (Honest State)

| Check | State | Notes |
|---|---|---|
| `py_compile` on `schema.py`, `runner.py`, `drive_bridge.py`, `test_all.py` | `done` (pass) | Executed after Milestone 10 edit |
| Offline `test_all.py` (schema, `tw2sp`/`s2twp`, whole-word glossary, personalization + plain fallback, strict refusal vs normal dialogue non-false-positive, `ui` empty-style/context, deduplication, resume, watch-chars, zero-compile prebuilt bundle + SHA-256 verification + zero `cmake`/`--rebuild-bin` assertions + `-lv 4` assertion, OS metadata ignore, `drive_bridge.py`, `.ipynb` JSON + `ast.parse` + `runner.log` + `DISCONNECT_ON_ERROR = False` + `IDLE_UNASSIGN_MINUTES`) | `done` (6/6 pass) | Executed after Milestone 10 edit using local mock SSE & HTTP servers and temp dirs |
| Code review, sanitization audit & Drive `scripts/hymt2/runner.py` SHA-256 sync | `done` (0 hits) | Verified zero secrets, zero local usernames/paths, zero copyrighted game strings, zero Docker CI triggers |
| Live Google Colab L4 GPU run (`hy_v3` load, `-fit on` VRAM margin, English refusal rate, H-scene quality, `s2twp` spot check) | `not tested` | Requires operator live Colab L4 run (`deploy_models_colab.ipynb` with `DEPLOY_HYMT2=True` then `hymt2_colab.ipynb`) |

---

## 5. First Live L4 Boot Checklist
1. Ensure 18.24 GB `Hy-MT2-30B-A3B-uncensored-heretic.i1-Q4_K_M.gguf` and `llama-server-l4-b12019.tar.gz` are deployed via `deploy_models_colab.ipynb` on Free CPU Colab.
2. Push a small ~50-line English adult-scene spot-check job with `drive_bridge.py push`.
3. Open `hymt2_colab.ipynb` on **Colab L4 GPU** (verify `DISCONNECT_ON_ERROR=False`, `IDLE_UNASSIGN_MINUTES=15`) and run `Ctrl+F9`.
4. After completion (or Gate 2 error), inspect `hymt2/work/llama.log` (`common_params_fit_impl` + `offloaded` + tok/s), `hymt2/output/<job>.qa_report.json`, and manually spot-check explicit tone and `s2twp` `干`/`發`/`後` conversions before running full game scripts.
