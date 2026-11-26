# 20 — Hy-MT2 Colab Workflow (`hymt2-heretic-colab`)

> **Branch:** `hymt2-heretic-colab` (branched from `origin/hymt2-heretic` at `b10f481`)
> **Updated:** 2026-11-26 12:33 +08:00
> **Agent:** Antigravity
> **Model:** Gemini 4 Neon
> **Status:** READY TO PUSH — Milestone 5 complete (zero-compile prebuilt CUDA 12.8 bundle on GitHub Release `colab-l4-llama-b12019`, all Colab compilation & `REBUILD_LLAMA_BIN` removed, `deploy_models_colab.ipynb` dual-SHA-256 binary deploy; offline tests 6/6 & sanitization passed)

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
Provide a foolproof **Google Colab L4 (24 GB) + Google Drive (5 TB) one-click workflow** for `Hy-MT2-30B-A3B-uncensored-heretic` (English -> Taiwan Traditional Chinese adult galgame H-scene translation), bypassing Vast.ai rental and `hanhua.py vast up` limitations while preserving all model/prompt decisions from `deploy/hymt2-heretic/`:
1. **Free CPU GGUF deployer** (`deploy_models_colab.ipynb`): mounts Drive, creates fixed folders, downloads pinned GGUF weights from HuggingFace (with `hf-mirror` fallback), verifies exact size and SHA-256 on both local SSD and Drive before atomic rename.
2. **Colab L4 one-click translator** (`hymt2_colab.ipynb` + `runner.py`): compiles/caches `llama-server` (`b12019`, `hy_v3` + `sm_89;80`) on Drive, stages GGUF to local SSD, enforces 4 hard GPU gates (L4/24GB+, `CUDA0`, full layer offload, `>= 18,000 MiB` VRAM, 3-line plain/explicit/SFX Gate `>= 8 tok/s`), runs 1-line-per-request 4-way parallel translation with `STYLE_ADULT` personalization prompt, `tw2sp` glossary/speaker, empty history for `ui`/`system`, plain-template mechanical fallback, OpenCC `t2s` canonical + `s2twp` default output, watch-char QA, `.DONE.json`, Drive flush, and `runtime.unassign()` in `finally`.
3. **Local Option A Drive Bridge** (`drive_bridge.py` + `skill/SKILL.md`): pure-stdlib CLI that auto-detects the single mounted Google Drive containing `galgame_mtl` (supports multi-account Drive e.g. `H:\`), checks `GoogleDriveFS.exe`, validates strict `.jsonl` + `glossary.json`, syncs `schema.py`/`runner.py` to Drive `scripts/hymt2/`, atomically pushes `<job>.jsonl` + `<job>.manifest.json`, checks status, and pulls verified outputs.
4. **Agent Migration SOP** (`AGENT_SOP_COLAB.md`): explicit migration guide for agents arriving with `vast.ai` handoff context plus EN->ZH adult H-scene specifics.

---

## 2. Locked Constants & Schema

| Item | Value |
|---|---|
| Colab Drive root | `/content/drive/MyDrive/galgame_mtl` |
| Model | `mradermacher/Hy-MT2-30B-A3B-uncensored-heretic-i1-GGUF` (`Hy-MT2-30B-A3B-uncensored-heretic.i1-Q4_K_M.gguf`) |
| Pinned HF rev | `862b7d66847b339c0594b8de39d38b035ee75f25` |
| Exact size | `18,236,703,488` bytes |
| Exact SHA-256 | `b0358d0319aae3f821fef4989bd85f409336139bfc7db59fd049848b9882321c` |
| Server | `llama.cpp` `b12019` (`llama-server`), `127.0.0.1:11434`, ctx `32768`, `-np 4` (`8192`/slot), `-ngl 999`, `--jinja`, `--device CUDA0 --cont-batching` |
| Sampling | `temperature=0.7`, `top_p=1.0`, `top_k=0`, `repeat_penalty=1.0`, `max_tokens=1024`, `stream=True` |
| Traditional conversion | Default OpenCC `s2twp` (per `19_galgame_image_branches.md`; `s2t` selectable via `--tw-mode`) |

---

## 3. Progress Log

| Timestamp (+08:00) | Milestone | Outcome | Notes |
|---|---|---|---|
| 2026-11-26 01:30 | Milestone 1: Branch & Core Implementation | Done | Created `deploy/colab-hymt2/` (`schema.py`, `runner.py`, `drive_bridge.py`, `deploy_models_colab.ipynb`, `hymt2_colab.ipynb`, `AGENT_SOP_COLAB.md`, `skill/SKILL.md`, `README.md`, `test_all.py`) and updated `llm_handoff/INDEX.md` |
| 2026-11-26 01:32 | Milestone 2: Code Review, Offline Tests & Sanitization | Done | Verified `b12019` `LLAMA_ARG_*` env vars, `tw2sp`/`s2twp` one-to-many watch-char behavior (`干/發/後`), `llama.log` fd closure, `try/except` around `drive.flush_and_unmount()`, and Drive glossary presence warning. `py_compile` + `test_all.py` (5/5 tests) passed; sanitization scan 0 hits |
| 2026-11-26 11:52 | Milestone 3: Colab Logging, CMake & Drive FUSE Parity Fixes | Done | Ported Milestone 3 fixes from `sakura-ollama-colab`: live `Popen` streaming + `hymt2/work/runner.log` tee + tail in exception; explicit `_find_nvcc()` + `-DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF`; early Gate 0 VRAM check; single-copy NVMe `stage_model` + SHA-256 verify; `copyfile` instead of `copy2` and `OSError`-safe `fsync` on Drive FUSE; OS metadata ignore in `input/`. `py_compile` + `test_all.py` (5/5) + sanitization (0 hits) passed |
| 2026-11-26 12:02 | Milestone 4: Disable `llama-ui` Prebuilt Fetch for `b12019` | Done | Ported Milestone 4 fix from `sakura-ollama-colab` (`-DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF` so shallow-clone `b1` does not pull mismatched `latest` UI assets). Re-ran `py_compile`, `test_all.py` (5/5), sanitization (0 hits) |
| 2026-11-26 12:33 | Milestone 5: Zero-Compile Prebuilt Bundle & Full Build Removal | Done | Eliminated ALL compilation from Colab: repacked official `ggml-org/llama.cpp` `b12019` Ubuntu x64 CUDA 12.8 `llama-server` + runtime `.so*` libs (`773,245,209` bytes, SHA-256 `802a169e33dde5e73403c2a36f3d6dbfd2d34d5e9ccd898e79a1eb899bcac947`, 43 entries) and published to GitHub Release `colab-l4-llama-b12019`. Removed `git clone`/`cmake`/`--rebuild-bin`/`REBUILD_LLAMA_BIN` from `runner.py` and `hymt2_colab.ipynb`; updated `deploy_models_colab.ipynb` to deploy the pinned tarball directly to Drive `bin/` with dual SHA-256 verification; updated `README.md` & `AGENT_SOP_COLAB.md`. Verified `py_compile`, `test_all.py` (6/6), sanitization (0 hits) |

---

## 4. Verification Status (Honest State)

| Check | State | Notes |
|---|---|---|
| `py_compile` on `schema.py`, `runner.py`, `drive_bridge.py`, `test_all.py` | `done` (pass) | Executed after Milestone 5 edit |
| Offline `test_all.py` (schema, `tw2sp`/`s2twp`, whole-word glossary, personalization + plain fallback, refusal, `ui` empty-style/context, deduplication, resume, watch-chars, zero-compile prebuilt bundle + SHA-256 verification + zero `cmake`/`--rebuild-bin` assertions, OS metadata ignore, `drive_bridge.py`, `.ipynb` JSON + `ast.parse` + `runner.log` + `DEPLOY_LLAMA_BIN`) | `done` (6/6 pass) | Executed after Milestone 5 edit using local mock SSE & HTTP servers and temp dirs |
| Code review & sanitization audit | `done` (0 hits) | Verified zero secrets, zero local usernames/paths, zero copyrighted game strings, zero Docker CI triggers |
| Live Google Colab L4 GPU run (`hy_v3` load, English refusal rate, H-scene quality) | `not tested` | Requires operator Colab account & Drive mount after branch push |
