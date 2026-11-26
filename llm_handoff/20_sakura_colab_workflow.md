# 20 — Sakura Colab Workflow (`sakura-ollama-colab`)

> **Branch:** `sakura-ollama-colab` (branched from `origin/sakura-ollama` at `a94c8fb`)
> **Updated:** 2026-11-26 14:30 +08:00
> **Agent:** Antigravity
> **Model:** Gemini 4 Neon
> **Status:** READY TO PUSH — Milestone 9 complete (fixed C9 `strict_line_hint` prompt echo on real Sakura-GalTransl-14B-v3.8, removed duplicate `LLAMA_ARG_LOG_VERBOSITY`, added asymmetric split-quote & heart-glyph normalization, changed `DISCONNECT_ON_ERROR` default to `False`, updated mock SSE strict prompt check; offline tests 6/6, 1.8k real-format stress test & sanitization passed)
> **Session Postmortem & Full Context:** See [21_colab_session_postmortem.md](21_colab_session_postmortem.md) for the complete C1–C9 failure table, 7-item operator accountability log, F1–F7 review fixes, and live Colab L4 specs.

---

## 1. Identity & Scope

| Field | Value |
|---|---|
| Agent harness | Antigravity |
| Model | Gemini 4 Neon |
| Branch | `sakura-ollama-colab` |
| Base commit | `a94c8fb` (`origin/sakura-ollama`) |
| Target directory | `deploy/colab-sakura/` |
| Touched existing files | `llm_handoff/INDEX.md` only (Dockerfiles and `.github/workflows/**` untouched) |

### Purpose
Replace the `vast.ai` rental workflow for the Japanese -> Traditional Chinese Galgame script translation branch (`sakura-ollama`) with a foolproof **Google Colab L4 (24 GB) + Google Drive (5 TB) one-click workflow**:
1. **Free CPU GGUF deployer** (`deploy_models_colab.ipynb`): mounts Drive, creates fixed folders, downloads pinned GGUF weights from HuggingFace (with `hf-mirror` fallback), verifies exact size and SHA-256 on both local SSD and Drive before atomic rename.
2. **Colab L4 one-click translator** (`sakura_colab.ipynb` + `runner.py`): compiles/caches `llama-server` (`b12019`, `sm_89;80`) on Drive, stages GGUF to local SSD, enforces 4 hard GPU gates (never silently falls back to CPU), runs 3-line pre-flight Gate + tok/s check, runs deduplicated 4-way parallel Sakura-GalTransl-14B-v3.8 batch-8 translation with checkpoint/resume, OpenCC `t2s`/`s2t` + watch-char/roundtrip QA, writes `.DONE.json`, flushes Drive, and calls `runtime.unassign()` in `finally`.
3. **Local Option A Drive Bridge** (`drive_bridge.py` + `skill/SKILL.md`): pure-stdlib CLI that auto-detects the single mounted Google Drive containing `galgame_mtl` (supports multi-account Drive e.g. `H:\`), checks `GoogleDriveFS.exe`, validates strict `.jsonl` + `glossary.json`, syncs `schema.py`/`runner.py` to Drive `scripts/sakura/`, atomically pushes `<job>.jsonl` + `<job>.manifest.json`, checks status, and pulls verified outputs after `.DONE.json` SHA-256 verification.
4. **Agent Migration SOP** (`AGENT_SOP_COLAB.md`): explicit migration guide for agents arriving with `vast.ai` handoff context.

---

## 2. Locked Constants & Schema

| Item | Value |
|---|---|
| Colab Drive root | `/content/drive/MyDrive/galgame_mtl` |
| Model | `SakuraLLM/Sakura-GalTransl-14B-v3.8` (`Sakura-Galtransl-14B-v3.8.gguf`, Q6_K) |
| Pinned HF rev | `10bab61b45d8970dc5c5fd7a4aec920d49a3cd8e` |
| Exact size | `12,124,681,536` bytes |
| Exact SHA-256 | `6e32fcf5bfdf78114745d2bf6bd87fb107ad7f68a611f4c2a886a4378846e334` |
| Server | `llama.cpp` `b12019` (`llama-server`), `127.0.0.1:11434`, ctx `16384`, `-np 4`, `-ngl 999`, `--device CUDA0 --cont-batching` |
| Sampling | `temperature=0.3` (retry `0.1`, gate `0.0`), `top_p=0.8`, `repeat_penalty=1.0`, `max_tokens=2048` (batch) / `512` (single), `stream=True` |
| Traditional conversion | Default OpenCC `s2t` (per operator historical rule; `s2twp` selectable via `--tw-mode`) |

---

## 3. Progress Log

| Timestamp (+08:00) | Milestone | Outcome | Notes |
|---|---|---|---|
| 2026-11-26 01:25 | Milestone 1: Branch & Core Implementation | Done | Created `deploy/colab-sakura/` (`schema.py`, `runner.py`, `drive_bridge.py`, `deploy_models_colab.ipynb`, `sakura_colab.ipynb`, `AGENT_SOP_COLAB.md`, `skill/SKILL.md`, `README.md`, `test_all.py`) and updated `llm_handoff/INDEX.md` |
| 2026-11-26 01:28 | Milestone 2: Code Review, Offline Tests & Sanitization | Done | Fixed 5 review items (closed `llama.log` parent fd, simplified fallback `build/bin` search, removed redundant ternary, added Drive glossary warning on `push` without `--glossary`, wrapped `drive.flush_and_unmount()` in `try/except` inside notebooks so `runtime.unassign()` always executes). `py_compile` + `test_all.py` (5/5 tests) passed; sanitization scan 0 hits |
| 2026-11-26 11:51 | Milestone 3: Live Colab Failure Fix | Done | First real Colab run exited rc=1 silently with empty `bin/` and `work/` (notebook `subprocess.run` hid child stdout/stderr and saved no `runner.log`, while `b12019` CMake had `LLAMA_OPENSSL=ON` + deprecated `LLAMA_CURL`). Fixed: live-streamed `Popen` + tee to `sakura/work/runner.log` + tail in exception; explicit `_find_nvcc()` + `-DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF`; Gate 0 VRAM check moved before build/staging; `stage_model` copies once to NVMe then verifies SHA-256; replaced `copy2` with `copyfile` and ignored `OSError` on `fsync` for Drive FUSE; ignored OS metadata files in `input/`. `py_compile` + `test_all.py` (5/5) + sanitization (0 hits) passed |
| 2026-11-26 12:01 | Milestone 4: Fix `b12019` `llama-ui-assets` CMake Failure | Done | Live Colab `runner.log` showed `cmake --build` failed at `llama-ui-assets` (`ui_validate_assets`: shallow `--depth 1` clone produced `b1`, fell back to `ggml-org/llama-ui/resolve/latest/dist.tar.gz`, mismatching `b12019` `bundle[hash].js` check). Verified in `b12019` `scripts/ui-assets.cmake` and added `-DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF` (`n_assets == 0` early return). Re-ran `py_compile`, `test_all.py` (5/5), sanitization (0 hits) |
| 2026-11-26 12:20 | Milestone 5: Prebuilt-First Zero-Compile `llama-server` & Free-CPU Pre-Caching | Done | Eliminated slow paid-L4 compilation by switching `ensure_llama_server` to 4 prebuilt tiers: (1A) exact env-hash Drive cache, (1B) compatible `llama-server-l4-b12019.tar.gz` / other Drive `bin/` bundle verified via `--version` + `--list-devices` CUDA check, (2A) project GitHub Release `colab-l4-llama-b12019` with SHA-256 verification, (2B) official `ggml-org/llama.cpp` `b12019` Ubuntu CUDA 12.8 + `cudart` prebuilt archives (exact byte size + SHA-256 verified) repacked to Drive `bin/`. Fail-fast error refuses silent source compile unless `--rebuild-bin` (`REBUILD_BIN=True`) is explicitly set. Added `DEPLOY_LLAMA_BIN = True` free-CPU pre-caching to `deploy_models_colab.ipynb`. Re-ran `py_compile`, `test_all.py` (6/6), sanitization (0 hits) |
| 2026-11-26 12:35 | Milestone 6: Complete Removal of Colab Build Code & Pinned Release Bundle | Done | Repacked official `ggml-org/llama.cpp` `b12019` Ubuntu x64 CUDA 12.8 `llama-server` + runtime `.so*` libs (`773,245,209` bytes, SHA-256 `802a169e33dde5e73403c2a36f3d6dbfd2d34d5e9ccd898e79a1eb899bcac947`, 43 entries) and published to GitHub Release `colab-l4-llama-b12019`. Removed ALL `git clone`/`cmake`/`--rebuild-bin`/`REBUILD_LLAMA_BIN` code from `runner.py` and `sakura_colab.ipynb`; updated `deploy_models_colab.ipynb` to deploy the pinned tarball directly to Drive `bin/` with dual SHA-256 verification; updated `README.md` & `AGENT_SOP_COLAB.md`. Re-ran `py_compile`, `test_all.py` (6/6), sanitization (0 hits) |
| 2026-11-26 13:21 | Milestone 7: Fix `b12019` Log Verbosity (`-lv 4`) for Gate 2 GPU Offload Verification | Done | Live Colab L4 run passed Gate 0 (`NVIDIA L4 23034 MiB`), Drive prebuilt extraction (`llama-server-l4-b12019.tar.gz`), 12.12 GB SSD staging + SHA-256, Gate 1 (`CUDA0`), and 4.6 s model load, then failed Gate 2 (`parse_gpu_offload_log`). Root cause verified in `llama.log` + `b12019` `common/log.cpp`: `GGML_LOG_LEVEL_INFO` maps to `LOG_LEVEL_TRACE = 4`, while default verbosity is `3`, suppressing `offloaded X/Y layers to GPU`. Added `"-lv", "4"` + `LLAMA_ARG_LOG_VERBOSITY=4`, clean `proc.terminate()` + `llama.log` tail on Gate 2 errors, and `0.3 -> 0.1` temperature retry per sample in `run_preflight_gate`. Re-ran `py_compile`, `test_all.py` (6/6), sanitization (0 hits) |
| 2026-11-26 13:35 | Milestone 8: Complete Code Review Fixes (F1, F2, F4, F5, F6, F7) | Done | Fixed all remaining review items prior to next live Colab L4 run: (F1) polled `llama.log` up to `10 x 0.5s` (`5s` max) in `start_and_verify_server` for `b12019` `common/log.cpp` async logger flush race; (F2) fixed `roundtrip_mismatches` tautology to compare `t2s(tw) != cn`; (F4) added `DISCONNECT_ON_ERROR = True` toggle in `sakura_colab.ipynb` so interactive watchers can preserve scarce L4 allocations on startup/gate errors; (F5) switched `drive_bridge.py` from `shutil.copy2` to `copyfile` for Drive FUSE safety and added informational `Prebuilt bin` readiness in `doctor`; (F6) fixed stale `building` comment in `main()`; (F7) moved `_log_gpu_progress` outside `ckpt_lock` under `gpu_log_lock`. Re-ran `py_compile`, `test_all.py` (6/6), sanitization (0 hits) |
| 2026-11-26 14:30 | Milestone 9: Fix Live Run #4 `strict_line_hint` Prompt Echo & Split-Quote/Default Fixes | Done | Live Colab L4 Run #4 passed Gate 0 (`NVIDIA L4 23034 MiB`), SSD copy (`249.2s`), Gate 1 (`CUDA0`), Gate 2 (`49/49 layers`), Gate 3 (`14378 MiB VRAM`), and real inference (`20.84 tok/s`), then failed Gate 4 sample 1 because `build_sakura_messages(..., strict_line_hint=True)` injected parenthetical line-count instructions after `：`, causing Sakura-GalTransl-14B-v3.8 to echo that line before the accurate translation. Removed `strict_line_hint` completely, added `(0.3, 0.1, 0.0)` gate fallback, removed redundant `LLAMA_ARG_LOG_VERBOSITY` env override, added asymmetric split-quote & heart-glyph normalization, switched `DISCONNECT_ON_ERROR` default to `False` in `sakura_colab.ipynb`, and tightened `test_all.py` mock server to HTTP 400 on any text between `：` and `\n`. Re-verified offline suite + 1,808-row stress test + sanitization (0 hits) |

---

## 4. Verification Status (Honest State)

| Check | State | Notes |
|---|---|---|
| `py_compile` on `schema.py`, `runner.py`, `drive_bridge.py`, `test_all.py` | `done` (pass) | Executed after Milestone 9 edit |
| Offline `test_all.py` (schema, BOM, normalization, split-quote/heart normalization, strict GalTransl prompt HTTP 400 check, mock SSE, `8->4->2->1` split retry, `ui` empty-history, deduplication, resume, `s2t` watch-chars & roundtrip, zero-compile prebuilt bundle + SHA-256 verification + zero `cmake`/`--rebuild-bin`, `-lv 4` assertion, OS metadata ignore, `drive_bridge.py`, `.ipynb` JSON + `ast.parse` + `runner.log` + `DISCONNECT_ON_ERROR = False`) | `done` (6/6 pass) | Executed after Milestone 9 edit using local mock SSE & HTTP servers and temp dirs |
| 1,808-record offline stress test (strict prompt guard + auto-closed asymmetric split quotes) | `done` (pass) | Executed against queued dataset in isolated temp dir (`missing=0`, `failures=0`, `roundtrip_mismatches=0`) |
| Code review & sanitization audit | `done` (0 hits) | Verified zero secrets, zero local usernames/paths, zero copyrighted game strings, zero Docker CI triggers |
| Live Colab L4 Gates 0–3 (`NVIDIA L4 23034 MiB`, `49/49 layers`, `14378 MiB VRAM`, `20.84 tok/s`) | `done` (pass on Run #4) | Verified on real Colab L4 Run #4 (`llama.log` + stdout) prior to Milestone 9 prompt fix |
| Full live Colab L4 Gate 4 + end-to-end translation run (`sakura_colab.ipynb` post-Milestone 9) | `not tested` | Requires operator Colab L4 rerun after Milestone 9 push |
