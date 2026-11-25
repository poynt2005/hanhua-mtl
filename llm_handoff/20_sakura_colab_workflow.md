# 20 — Sakura Colab Workflow (`sakura-ollama-colab`)

> **Branch:** `sakura-ollama-colab` (branched from `origin/sakura-ollama` at `a94c8fb`)
> **Updated:** 2026-11-26 01:28 +08:00
> **Agent:** Antigravity
> **Model:** Gemini 4 Neon
> **Status:** READY TO PUSH — Milestone 2 complete (code review, offline tests, and sanitization audit passed; live Colab L4 GPU run pending operator)

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
| Sampling | `temperature=0.3` (retry `0.1`), `top_p=0.8`, `repeat_penalty=1.0`, `max_tokens=2048` (batch) / `512` (single), `stream=True` |
| Traditional conversion | Default OpenCC `s2t` (per operator historical rule; `s2twp` selectable via `--tw-mode`) |

---

## 3. Progress Log

| Timestamp (+08:00) | Milestone | Outcome | Notes |
|---|---|---|---|
| 2026-11-26 01:25 | Milestone 1: Branch & Core Implementation | Done | Created `deploy/colab-sakura/` (`schema.py`, `runner.py`, `drive_bridge.py`, `deploy_models_colab.ipynb`, `sakura_colab.ipynb`, `AGENT_SOP_COLAB.md`, `skill/SKILL.md`, `README.md`, `test_all.py`) and updated `llm_handoff/INDEX.md` |
| 2026-11-26 01:28 | Milestone 2: Code Review, Offline Tests & Sanitization | Done | Fixed 5 review items (closed `llama.log` parent fd, simplified fallback `build/bin` search, removed redundant ternary, added Drive glossary warning on `push` without `--glossary`, wrapped `drive.flush_and_unmount()` in `try/except` inside notebooks so `runtime.unassign()` always executes). `py_compile` + `test_all.py` (5/5 tests) passed; sanitization scan 0 hits |

---

## 4. Verification Status (Honest State)

| Check | State | Notes |
|---|---|---|
| `py_compile` on `schema.py`, `runner.py`, `drive_bridge.py`, `test_all.py` | `done` (pass) | Executed after final edit |
| Offline `test_all.py` (schema, BOM, normalization, mock SSE, `8->4->2->1` split retry, `ui` empty-history, deduplication, resume, `s2t` watch-chars & roundtrip, `drive_bridge.py`, `.ipynb` JSON + `ast.parse`) | `done` (5/5 pass) | Executed after final edit using local mock SSE server (`2180.5 tok/s` mock) and temp dirs |
| Code review & sanitization audit | `done` (0 hits) | Verified zero secrets, zero local usernames/paths, zero copyrighted game strings, zero Docker CI triggers |
| Live Google Colab L4 GPU run (`sakura_colab.ipynb` & `deploy_models_colab.ipynb`) | `not tested` | Requires operator Colab account & Drive mount after branch push |
