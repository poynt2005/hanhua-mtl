# 21 — Colab Migration Session Postmortem & Complete Context Handoff

> **Date:** 2026-11-28 (+08:00)
> **Agent:** Antigravity
> **Model:** Gemini 4 Neon
> **Branches covered:** `sakura-ollama-colab` (from `origin/sakura-ollama` `a94c8fb`) and `hymt2-heretic-colab` (from `origin/hymt2-heretic` `b10f481`)
> **Scope:** Complete session history from initial Colab migration through Milestone 11 (`sakura-ollama-colab`) and Milestone 10 (`hymt2-heretic-colab`)

---

## 1. Full Session Context & Evolution

### 1.1 Why Colab L4 (24 GB VRAM) + Google Drive (5 TB) Replaced `vast.ai`
On the two galgame script translation branches (`sakura-ollama` and `hymt2-heretic`), `vast.ai` per-hour GPU rentals introduced operational overhead, port-forward timeout drops, and real-money thrashing risk during debugging. Migrating these two branches onto **Google Colab L4 (24 GB VRAM) + Google Drive (5 TB)** solves those issues while isolating all changes strictly to `deploy/colab-sakura/`, `deploy/colab-hymt2/`, and branch-scoped `llm_handoff/` docs:
- `master`, `sakura-ollama`, `hymt2-heretic`, all `Dockerfile`s, and `.github/workflows/**` remain **untouched** (zero Docker CI triggers).
- **Split lifecycle:**
  1. **Free CPU GGUF & Binary Deployer (`deploy_models_colab.ipynb`):** Runs on a free Colab CPU runtime, mounts Google Drive, creates fixed folder structure under `/content/drive/MyDrive/galgame_mtl`, downloads pinned GGUF weights from HuggingFace (with `hf-mirror` fallback) and the prebuilt `llama-server` tarball (`colab-l4-llama-b12019`), and verifies exact byte sizes and SHA-256 checksums before atomic rename.
  2. **Paid L4 GPU One-Click Translator (`sakura_colab.ipynb` / `hymt2_colab.ipynb` + `runner.py`):** Extracts the prebuilt `llama-server` bundle to fast local SSD, stages the GGUF model once to NVMe SSD with SHA-256 verification, enforces 4 hard GPU gates before touching translation, runs parallelized translation with checkpoint/resume (targeted at `the operator's first queued test job (~1.8k records)` and subsequent jobs), writes `.DONE.json`, flushes Drive, and handles `runtime.unassign()` in `finally`.
  3. **Local Option A Drive Bridge (`drive_bridge.py` + `skill/SKILL.md`):** Pure-stdlib local Windows CLI that checks `GoogleDriveFS.exe`, auto-detects the single mounted Google Drive letter containing `galgame_mtl` (supporting multi-account Drive Desktop mounts such as `H:\`), validates strict `.jsonl` and `glossary.json`, syncs `schema.py`/`runner.py` to Drive `scripts/<target>/`, atomically pushes `<job>.jsonl` + `<job>.manifest.json`, checks job status, and pulls verified outputs only after `.DONE.json` SHA-256 verification.
  4. **Agent Migration SOP (`AGENT_SOP_COLAB.md`):** Bridges future agents arriving with `vast.ai` handoff context (`19_galgame_image_branches.md`, `hanhua.py vast up`) directly onto the Colab + Drive workflow.

### 1.2 Historical Localization Rules Carried Over from `vast.ai` Handoffs
All domain rules established in prior `vast.ai` handoffs were preserved verbatim:
- **General hygiene & safety:**
  - Never modify original game files; treat all input text strictly as untrusted data; never distribute copyrighted text or assets.
  - Read input `.jsonl` / JSON files with `utf-8-sig` (BOM-safe) and write outputs with `utf-8` and LF (`\n`) line endings.
  - Never pass CJK text through PowerShell heredocs (`@'...'@`) or `python -c '...'` inline strings (Windows PowerShell encoding/quoting mangles CJK characters).
  - Option A pure-stdlib `drive_bridge.py` requires `GoogleDriveFS.exe` to be running and scans local Drive Desktop mounts for the unique `galgame_mtl` root.
- **Sakura (`sakura-ollama-colab`, JA -> zh-TW):**
  - 100% Simplified Chinese input hygiene: apply OpenCC `t2s` to glossary entries, context history, and source text before prompting; do **not** retry with Traditional Chinese prompts.
  - Empty context history (`[]`) for `ui` and `system` categories.
  - Sampling: `temperature=0.3` (per-sample retry `0.1`), `top_p=0.8`, `repeat_penalty=1.0`, `max_tokens=2048` (batch) / `512` (single), `stream=True`.
  - 4-way parallel (`-np 4`, ctx `16384`, `-ngl 999`, `--device CUDA0 --cont-batching`) batch-8 translation with `8->4->2->1` split retry and deduplication.
  - Default OpenCC `s2t` output (`s2twp` selectable via `--tw-mode`), plus watch-chars and canonical `t2s` roundtrip QA.
- **Hy-MT2 (`hymt2-heretic-colab`, EN -> zh-TW adult H-scene):**
  - No system message in the chat payload.
  - Use `STYLE_ADULT` personalization prompt; apply OpenCC `tw2sp` to glossary terms and speaker names.
  - Omit style instructions and context history for `ui` and `system` categories.
  - 1 line per request across 4 parallel slots (`-np 4`, ctx `32768` / `8192` per slot, `-ngl 999`, `--jinja`, `--device CUDA0 --cont-batching`, `temperature=0.7`, `top_p=1.0`, `top_k=0`, `repeat_penalty=1.0`, `max_tokens=1024`, `stream=True`).
  - Mechanical plain-template fallback when personalized/styled prompting hits a refusal or format error.
  - Default OpenCC `s2twp` output (`s2t` selectable via `--tw-mode`), canonical `t2s` normalization, and watch-chars QA (accounting for `tw2sp`/`s2twp` one-to-many characters such as `干`/`發`/`後`).

### 1.3 Locked Artifacts & Observed Live Colab L4 Specs

| Artifact / Spec | Locked Value |
|---|---|
| **Sakura GGUF** | `SakuraLLM/Sakura-GalTransl-14B-v3.8` (`Sakura-Galtransl-14B-v3.8.gguf`, Q6_K — max quantization available on HF), rev `10bab61b45d8970dc5c5fd7a4aec920d49a3cd8e`, `12,124,681,536` bytes, SHA-256 `6e32fcf5bfdf78114745d2bf6bd87fb107ad7f68a611f4c2a886a4378846e334` |
| **Hy-MT2 GGUF** | `mradermacher/Hy-MT2-30B-A3B-uncensored-heretic-i1-GGUF` (`Hy-MT2-30B-A3B-uncensored-heretic.i1-Q4_K_M.gguf`), rev `862b7d66847b339c0594b8de39d38b035ee75f25`, `18,236,703,488` bytes, SHA-256 `b0358d0319aae3f821fef4989bd85f409336139bfc7db59fd049848b9882321c` |
| **Prebuilt L4 `llama-server` bundle** | GitHub Release `poynt2005/hanhua-mtl` tag `colab-l4-llama-b12019`, asset `llama-server-l4-b12019.tar.gz`, `773,245,209` bytes, SHA-256 `802a169e33dde5e73403c2a36f3d6dbfd2d34d5e9ccd898e79a1eb899bcac947`, `43` entries (repacked via POSIX `tarfile` from official `ggml-org/llama.cpp` `b12019` `llama-b12019-bin-ubuntu-cuda-12.8-x64.tar.gz` + `cudart-llama-b12019-bin-ubuntu-cuda-12.8-x64.tar.gz` — zero compilation) |
| **Observed live Colab L4 VM** | `NVIDIA L4` (`23034 MiB` VRAM, `sm_89`), Ubuntu 24.04 (GCC 13.3.0 / GLIBC 2.39), Python 3.13, CUDA 13.0.88, 12 CPU cores |
| **Observed live Colab timings & Run #4/#5 metrics** | 12.12 GB Drive -> local NVMe SSD copy: `249.2s`–`381.9s`; L4 GPU model load: `4.6s`; Gate 2: `49/49` layers offloaded (`10948 MiB` weights + `3072 MiB` KV + compute); Gate 3 VRAM used: `14378 MiB` (peak active `14388 MiB`); 4-way Gate 4 speed: `24.1–24.2 tok/s` at 92–100% GPU util (~71–75W) |

---

## 2. Exhaustive Pitfall & Failure Postmortem Table (C1–C11)

| ID | Phase / Event | What Failed & Root Cause | Resolution & Permanent Fix |
|---|---|---|---|
| **C1** | Session start | Agent began branching/editing before answering the operator's SOP question directly. | Hard stop and explicit apology; rule added: always answer the operator's question completely before touching git branches or files. |
| **C2** | Local Windows shell execution | PowerShell 5.1 rejected `&&` chaining, stripped quotes in `python -c '...'`, and risked CJK encoding corruption in inline heredocs. | Use PowerShell `;` or `$LASTEXITCODE` checks, run standalone `.py` scripts/files instead of inline CJK heredocs, and enforce `utf-8-sig` read / `utf-8` LF write in code. |
| **C3** | Offline `test_all.py` initial run | Initial test assertions had two bugs: a placeholder-reorder test expectation mismatch and overlooking OpenCC `s2twp` phrase-level mapping differences vs `s2t`. | Corrected test assertions to match real OpenCC `s2t` vs `s2twp` behavior (including one-to-many watch characters like `干`/`發`/`後`) and verified 5/5 then 6/6 offline tests. |
| **C4** | Live Colab Run #1 (silent `rc=1` black box) | Notebook `subprocess.run` swallowed child stdout/stderr and exited `rc=1` before saving `runner.log`, then `finally` called `runtime.unassign()`, wiping the VM and all diagnostic evidence. Additional latent bugs surfaced: Gate 0 ran too late (after build/staging), `b12019` CMake defaulted `LLAMA_OPENSSL=ON` (with deprecated `LLAMA_CURL`), `stage_model` read the 12 GB GGUF twice over Drive FUSE, `shutil.copy2` and `os.fsync` were fragile on Drive FUSE, and OS metadata files in `input/` could break job discovery. | Streamed child `Popen` output live and teed it to `<target>/work/runner.log` (with log tail included in raised exceptions); moved Gate 0 VRAM check before staging; turned off `LLAMA_OPENSSL` (later replaced entirely by prebuilt binary in C6); copied GGUF once to local NVMe then verified SHA-256; switched to `copyfile` + `OSError`-safe `fsync`; ignored OS metadata files in `input/`. |
| **C5** | Live Colab Run #2 (`llama-ui-assets` CMake failure) | Shallow `git clone --depth 1` of `llama.cpp` `b12019` resolved tag/describe to `b1`; `scripts/ui-assets.cmake` hit HTTP 404 on HuggingFace `b1`, fell back to `latest`, and failed `ui_validate_assets` against `b12019`'s SvelteKit `bundle[hash].js` check. | Verified in `b12019` `scripts/ui-assets.cmake` and added `-DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF` (`n_assets == 0` early return) before eliminating source compilation entirely in C6. |
| **C6** | Architectural mistake: compiling `llama.cpp` on paid Colab L4 | Agent wrote source-compilation logic (`git clone` + `cmake`) on paid Colab L4 units instead of checking official `ggml-org/llama.cpp` prebuilt binaries — even though the `vast.ai` Dockerfiles already used prebuilt `ollama/ollama:latest` and `ghcr.io/ggml-org/llama.cpp:server-cuda-b12019`. This burned operator compute units, and the agent briefly compounded the mistake by proposing compiling on `an external free cloud Linux host` until the operator called out why upstream prebuilt binaries weren't used from the start. | Repacked official `ggml-org/llama.cpp` `b12019` Ubuntu CUDA 12.8 binaries + `cudart` runtime `.so*` libs (`773,245,209` bytes, SHA-256 `802a169e33dde5e73403c2a36f3d6dbfd2d34d5e9ccd898e79a1eb899bcac947`, 43 entries) into GitHub Release `poynt2005/hanhua-mtl` tag `colab-l4-llama-b12019`. Deleted **all** `git clone`/`cmake`/`--rebuild-bin`/`REBUILD_LLAMA_BIN` code from `runner.py` and both notebooks, and updated `deploy_models_colab.ipynb` to pre-deploy the tarball to Drive `bin/` on a free CPU runtime. |
| **C7** | Live Colab Run #3 (Gate 2 false-kill + scarce L4 loss) | Live Colab L4 passed Gate 0 (`NVIDIA L4 23034 MiB`), prebuilt `bin/` extraction, 12.12 GB SSD copy (`289.4s`), Gate 1 (`CUDA0`), and `4.6s` GPU model load, then failed Gate 2 (`parse_gpu_offload_log`). Root cause in `llama.log` + `b12019` `common/log.cpp`: `GGML_LOG_LEVEL_INFO` maps to `LOG_LEVEL_TRACE = 4`, while default verbosity is `3`, suppressing `offloaded X/Y layers to GPU`. Unconditional `AUTO_DISCONNECT=True` immediately released a scarce L4 VM. | Added `"-lv", "4"` + `LLAMA_ARG_LOG_VERBOSITY=4`, clean `proc.terminate()` + `llama.log` tail on Gate 2 failure, `0.3 -> 0.1` temperature retry per sample in Sakura `run_preflight_gate`, and later added `DISCONNECT_ON_ERROR` toggle (F4). |
| **C8** | Comprehensive code review pass before Run #4 | Pre-run audit uncovered 7 remaining correctness/robustness issues (`F1`–`F7`), including an async `common/log.cpp` flush race, a Sakura `roundtrip_mismatches` tautology, missing Hy-MT2 preflight fallback, Drive FUSE per-line checkpoint churn, and unconditional error disconnect. | Fixed all 7 findings (`F1`–`F7`, detailed in §4) in Milestone 8 (`sakura-ollama-colab`) and Milestone 7 (`hymt2-heretic-colab`) and re-verified `py_compile`, `test_all.py` (6/6), and sanitization (0 hits). |
| **C9** | Live Colab Run #4 (Gate 4 `strict_line_hint` echo + old notebook `DISCONNECT_ON_ERROR=True`) | Run #4 passed Gate 0 (`NVIDIA L4 23034 MiB`), SSD staging (`249.2s`), Gate 1 (`CUDA0`), Gate 2 (`49/49 layers`), and Gate 3 (`14378 MiB VRAM`, `llama.log` `20.84 tok/s`), then failed Gate 4 sample 1 with 2 lines: `'（请严格输出恰好 1 行，一一对应，不要合并或拆分行）\n唯：早上好。睡得好吗？'`. Root cause: `build_sakura_messages(..., strict_line_hint=True)` injected that parenthetical instruction directly after `将下面的文本从日文翻译成简体中文：`, whereas Sakura-GalTransl-14B-v3.8 is fine-tuned to treat everything after `：` as source input, so it echoed that line before producing the accurate translation; `test_all.py` `_make_handler` had masked it via `after.split("\n", 1)[1]`. Additionally, `DISCONNECT_ON_ERROR` defaulted to `True` in Run #4 so the operator's L4 session disconnected immediately; audit also found `LLAMA_ARG_LOG_VERBOSITY=4` alongside `-lv 4` emitted a harmless `llama.log` overwrite warning and real queued data contains 12 asymmetric split-quote lines (`open != close`) that could be rejected if the model closes/opens quotes. | Removed `strict_line_hint` completely so `build_sakura_messages` always emits exact official GalTransl v3.8 prompt syntax (`：\n`); added `(0.3, 0.1, 0.0)` gate retry; removed redundant `LLAMA_ARG_LOG_VERBOSITY` env var (keeping `-lv 4`); added asymmetric split-quote (`trim_close_quote`/`trim_open_quote`) and exclusive `♥`<->`♡` normalization in `normalize_output_line`; defaulted `DISCONNECT_ON_ERROR = False` in both `sakura_colab.ipynb` and `hymt2_colab.ipynb`; hardened `test_all.py` mock handler to HTTP 400 if any text appears between `：` and `\n`. |
| **C10** | Large 14,269-row production run (`2,235` unique units) exposed 20 edge-case failures + 12 language-QA gaps | Run #5 succeeded on the first 1,808-row job (`24.2 tok/s`), but a 14,269-row production job (`2,235` unique units) finished `2,215` units with `20` hard failures (`46` missing rows) and `12` `no_cjk` / untranslated English rows not caught by existing checks: (1) sequential `wid` shifted whenever overrides were added, preventing safe incremental repair; (2) `repair_quotes` missed mid-sentence non-corner quotes, missing outer quote pairs, and half-width/curly quotes; (3) kana/loanword proper nouns hit `residual_kana` with no `overrides.jsonl` pass-through or glossary substitution path; (4) `{{TAG_n}}` drops lacked normalized `⟦0⟧` encoding and targeted system-prompt retries; (5) `\n` multiline units were prompted whole instead of segment-split and rejoined; (6) no `no_cjk`/`latin_only` QA gate existed to catch English loanword outputs or truncated text (`　」`); (7) `DISCONNECT_ON_ERROR=False` could leave an L4 idle without a bounded timer or explicit `STOP THE RUNTIME NOW` warning; (8) `failures.jsonl` lacked `unit_key`, `attempts`, `expected_placeholders`/`observed_placeholders`, and `suggested_fix`. | Implemented SHA-1 `unit_key` + legacy `wid` migration, generalized `repair_quotes`, internal `⟦0⟧` placeholder encode/decode, `overrides.jsonl` + glossary kana substitution, `\n` segment-split multiline handling, `no_cjk`/`latin_only` QA + verbatim command allowlist, `--repair-only`/`--job`/`--overrides` incremental repair, actionable `failures.jsonl`, and `IDLE_UNASSIGN_MINUTES=15` + `STOP THE RUNTIME NOW` warning in both `sakura_colab.ipynb` and `hymt2_colab.ipynb`. Verified with 15 offline tests + isolated 14,269-row repair dry-run (`cached=2215 | todo=20`). |
| **C11** | Comprehensive regex & validator audit against all 14,269 real production rows + `hymt2_prompt.py` | Auditing every regex across both branches against real repaired data revealed 4 false-positive / fidelity bugs: (1) Sakura `_KANA_RE` (`\u30a0-\u30ff`) included non-letter punctuation (`・` `U+30FB`, `ー` `U+30FC`, `゠` `U+30A0`), falsely failing 9 valid Chinese translations (e.g. `HP・MP`) as `residual_kana`; (2) Sakura `normalize_output_line` collapsed multiple leading `\u3000` full-width indents down to 1 across 5 multi-indent rows; (3) Sakura `repair_quotes` lacked Chinese `《》` / `【】` title-quote repairs; (4) Hy-MT2 applied broad refusal words (`抱歉`/`无法`/`I'm sorry`/`I can't`) to normal dialogue in `mechanical_problems()` instead of restricting broad refusal matching to Gate 4 (`_GATE_REFUSAL_RE`). | Narrowed Sakura `_KANA_RE` & `_JA_PRESENT_RE` to real kana letters, preserved exact leading `\u3000` count, added `《》` / `【】` quote repair (verified 0 errors and 0 unintended mutations across all 14,269 real rows), and separated Hy-MT2 `_GATE_REFUSAL_RE` from strict AI-boilerplate `_REFUSAL_RE` in `mechanical_problems()`. |

---

## 3. Operator Frustration & Accountability Log

This section records honestly all 9 operator frustration incidents from this session so future agents understand the exact mistakes made, why they cost time and compute units, and the non-negotiable guardrails required going forward.

| # | Incident | Root Cause & Impact |
|---|---|---|
| **1** | **Branching and editing before answering the operator's question** | The agent started creating branches/files instead of directly answering the operator's SOP question first, forcing the operator to intervene. |
| **2** | **Live Colab Run #1 failed silently with zero visible log (`runner.log` missing) and wiped the VM** | Notebook `subprocess.run` buffered/hid output and `finally: runtime.unassign()` disconnected the VM immediately, leaving the operator with zero diagnostic trace (`rc=1`, empty `work/`). |
| **3** | **Compiling `llama.cpp` from source on paid Colab L4 compute units** | Instead of checking upstream prebuilt artifacts first, the agent ran CMake + `nvcc` compilation on paid L4 runtime, wasting operator compute units on `C4` and `C5` build failures. |
| **4** | **Proposing `an external free cloud Linux host` compilation before noticing upstream prebuilt binaries** | Even though `vast.ai` Dockerfiles never compiled from source (`ollama/ollama:latest` and `ghcr.io/ggml-org/llama.cpp:server-cuda-b12019`), the agent suggested compiling on `an external free cloud Linux host` instead of using official `ggml-org/llama.cpp` `b12019` releases, forcing the operator to challenge why any build was needed at all and call out accountability for the wasted compute units. |
| **5** | **Operator had to ask twice whether agent skill/SOP docs needed updating** | While the 737 MB prebuilt bundle upload was running, the operator had to prompt twice before the agent addressed updating the workflow/skill/SOP documentation to match the zero-compile architecture. |
| **6** | **Live Colab Run #3 lost a scarce L4 right after 12.12 GB copy + 4.6 s model load** | Because `parse_gpu_offload_log` assumed `b12019` printed `offloaded X/Y layers to GPU` at default verbosity without checking `b12019` `common/log.cpp` (`LOG_LEVEL_TRACE = 4` vs default `3`), Gate 2 falsely killed a healthy GPU-loaded server and `runtime.unassign()` immediately dropped a scarce L4 GPU allocation. |
| **7** | **Live Colab Run #4 still hit Gate 4 `strict_line_hint` echo and disconnected another L4 VM** | After fixing Gate 2 (`-lv 4`), Run #4 passed Gates 0-3 on real L4 (`49/49` layers, `14378 MiB`), only to fail Gate 4 because `runner.py` injected `strict_line_hint` after `：`, violating GalTransl v3.8's trained prompt format while the mock handler's `split("\n", 1)[1]` masked the deviation offline; meanwhile `DISCONNECT_ON_ERROR` defaulted to `True` in the notebook so failing disconnected the L4 again. |
| **8** | **Large 14,269-row production run left 20 unrepairable failures + 12 undetected `no_cjk` / English outputs** | Although Run #5 succeeded cleanly on the 1,808-row validation job, running a 14,269-row (`2,235` unique-unit) production job via a downstream agent exposed missing deterministic quote/placeholder/multiline repairs, sequential `wid` cache invalidation preventing incremental `--repair-only` runs, no `no_cjk`/`latin_only` QA gate, and no bounded `IDLE_UNASSIGN_MINUTES` timer when `DISCONNECT_ON_ERROR=False`. |
| **9** | **Auditing C10 against 14,269 real production rows uncovered 4 more preventable false-positive bugs** | Testing the C10 runner directly against all 14,269 repaired production rows and `hymt2_prompt.py` revealed `_KANA_RE` flagging `・` (`U+30FB`) in valid Chinese (`9` of the `20` production failures were actually false-positive `・` lines), multi-`\u3000` indents collapsing to `1` across `60` rows, missing `《》`/`【】` title-quote repairs, and Hy-MT2 `_REFUSAL_RE` flagging normal dialogue words (`抱歉`/`无法`) — eroding operator trust due to incomplete real-data validation earlier. |

### Permanent Non-Negotiable Agent Guardrails
1. **Answer questions first:** Never branch, edit files, or run mutations before directly answering an operator's pending question.
2. **Prebuilt-first, zero-compile on paid GPUs:** Always inspect upstream official releases and existing `Dockerfile`s first. Never compile C++/CUDA engines on paid Colab/cloud GPUs when pinned upstream binaries exist.
3. **Never let a run die silently:** Always tee subprocess stdout/stderr live to a persistent log file (`runner.log`) and surface log tails in exceptions before any teardown runs.
4. **Verify exact pinned upstream behavior before gating:** Check the actual pinned version's source/CLI flags (e.g., `b12019` `common/log.cpp` `-lv 4` and `scripts/ui-assets.cmake`) against real behavior rather than assuming log strings or CMake defaults.
5. **Protect scarce GPU allocations with bounded timers:** Default `DISCONNECT_ON_ERROR = False` in notebooks for interactive debugging, ALWAYS flush Drive first, print `STOP THE RUNTIME NOW` with idle cost, and enforce `IDLE_UNASSIGN_MINUTES` (default `15`) so an idle runtime can never bill indefinitely.
6. **Keep skills, SOPs, and handoff docs synchronized immediately:** Whenever architecture or tooling changes, update `AGENT_SOP_COLAB.md`, `skill/SKILL.md`, `README.md`, and `llm_handoff/` without waiting to be asked twice.
7. **Never tamper with fine-tuned prompt triggers and never write permissive mocks:** Keep `将下面的文本从日文翻译成简体中文：\n` verbatim with zero injected text before `\n`; put retry guidance only in `system` role and test mocks must reject prompt deviations with HTTP 400.
8. **Audit every regex, normalizer, and validator against real production data fixtures:** Explicitly verify Unicode punctuation boundaries (such as `U+30FB`/`U+30FC` inside `U+30A0..U+30FF`), idempotency of normalizers, and dialogue non-false-positives against real datasets before shipping.

---

## 4. Code Review Findings & Fixes (F1–F7)

Completed in **Milestone 8** (`sakura-ollama-colab`) and **Milestone 7** (`hymt2-heretic-colab`) prior to the next live Colab L4 run:

| ID | Severity | Scope | Finding & Root Cause | Fix Applied |
|---|---|---|---|---|
| **F1** | **P1** | Both branches (`runner.py`) | `llama.cpp` `b12019` `common/log.cpp` uses an async logger thread; `/health` returning `200` can race ahead of the logger thread flushing `offloaded X/Y layers to GPU` to `llama.log`. | Updated `start_and_verify_server` on both branches to poll `llama.log` up to `10 x 0.5s` (`5s` max) until the offload line appears (or fail cleanly with log tail). |
| **F2** | **P1** | Sakura (`runner.py`) | `roundtrip_mismatches` had a tautology bug comparing `t2s(tw) != t2s(s2t(t2s(tw)))` instead of comparing against the canonical Simplified source `cn`. | Fixed the comparison to `t2s(tw) != cn` so genuine roundtrip drift from `cn` is accurately counted. |
| **F3** | **P2** | Hy-MT2 (`runner.py`) | `run_preflight_gate` lacked the mechanical plain-template fallback already implemented in `_translate_one_unit`, risking false preflight failures on styled prompts. | Added the plain-template mechanical fallback path to `run_preflight_gate` to match `_translate_one_unit`. |
| **F4** | **P2** | Both branches (`sakura_colab.ipynb`, `hymt2_colab.ipynb`) | Unconditional disconnect on startup/gate failure in `finally` immediately forfeited scarce L4 allocations and ~5 minutes of 12–18 GB SSD staging. | Added `DISCONNECT_ON_ERROR = True` toggle in both notebooks so operators watching interactively can set it to `False` to preserve the L4 VM on errors. |
| **F5** | **P2** | Both branches (`drive_bridge.py`) | `drive_bridge.py` still used `shutil.copy2` (which attempts metadata/timestamp preservation problematic on Google Drive FUSE/Desktop), and `doctor` did not report prebuilt `bin/` bundle readiness. | Switched `drive_bridge.py` to `shutil.copyfile` and added an informational `Prebuilt bin` readiness line in `doctor`. |
| **F6** | **P3** | Both branches (`runner.py`) | Hy-MT2 checkpointing opened and flushed Drive FUSE once per translated line (~1,800 opens for `the operator's first queued test job (~1.8k records)`), adding FUSE latency and churn; Sakura `main()` still had a stale `building` comment. | Updated Hy-MT2 `_run_queue` to batch-flush checkpoints every `10` units plus a guaranteed `finally` flush; removed the stale `building` comment in Sakura `main()`. |
| **F7** | **P3** | Both branches (`runner.py`) | `_log_gpu_progress` executed while holding `ckpt_lock`, serializing GPU telemetry logging behind checkpoint I/O. | Moved `_log_gpu_progress` outside `ckpt_lock` under a dedicated `gpu_log_lock`. |

---

## 5. Current Readiness & Verification Matrix

| Check / Stage | `sakura-ollama-colab` (Milestone 11) | `hymt2-heretic-colab` (Milestone 10) | Notes |
|---|---|---|---|
| `py_compile` (`schema.py`, `runner.py`, `drive_bridge.py`, `test_all.py`) | `done` (pass) | `done` (pass) | Verified after C11 edits |
| `test_all.py` (offline suite: 15 tests on Sakura; 6 tests on Hy-MT2) | `done` (15/15 pass) | `done` (6/6 pass) | Includes `・`/`ー` non-kana, multi-`\u3000` indent, `《》`/`【】` quotes, and Hy-MT2 normal-dialogue refusal non-false-positive |
| 14,269-row real repaired output audit (`validate_unit_translation` + `normalize_output_line`) | `done` (pass) | `n/a` | 0 errors and 0 unintended mutations across all 14,269 real rows |
| Sanitization scan (zero secrets, local paths, forbidden strings, game text) | `done` (0 hits) | `done` (0 hits) | Verified across repository deliverables and handoff docs |
| GitHub Release `colab-l4-llama-b12019` (`llama-server-l4-b12019.tar.gz`) | `done` (HTTP 200) | `done` (HTTP 200) | `773,245,209` bytes, SHA-256 `802a169e33dde5e73403c2a36f3d6dbfd2d34d5e9ccd898e79a1eb899bcac947`, 43 entries |
| Live Colab L4 Gates 0-4 + 1,808-row run (`24.2 tok/s`, 92-100% GPU util) | `done` (pass on Run #5) | `not tested` | Sakura Run #5 completed full 1,808-row pipeline; 14k production run surfaced C10 & C11 |
| Live Colab L4 `--repair-only` / Milestone 11 & 10 post-C11 runs | `not tested` | `not tested` | Requires operator live Colab L4 run after push |

---

## 6. Architecture Snapshot & Immediate Next Steps

### 6.1 Branch Isolation & Deliverables
- **`sakura-ollama-colab`** (`deploy/colab-sakura/`): `schema.py`, `runner.py`, `drive_bridge.py`, `deploy_models_colab.ipynb`, `sakura_colab.ipynb`, `AGENT_SOP_COLAB.md`, `skill/SKILL.md`, `README.md`, `test_all.py`.
- **`hymt2-heretic-colab`** (`deploy/colab-hymt2/`): `schema.py`, `runner.py`, `drive_bridge.py`, `deploy_models_colab.ipynb`, `hymt2_colab.ipynb`, `AGENT_SOP_COLAB.md`, `skill/SKILL.md`, `README.md`, `test_all.py`.
- **`master`**, **`sakura-ollama`**, **`hymt2-heretic`**, all `Dockerfile`s, and `.github/workflows/**` remain completely untouched.

### 6.2 Key Pending Live Checks (`not tested` on real Colab L4 GPU)
1. **Sakura post-C11 live Colab L4 run:** First real Colab L4 run using `--repair-only` / `<job>.overrides.jsonl` and post-C11 regex/indent normalization.
2. **Hy-MT2 first live Colab L4 boot (`deploy/colab-hymt2/`):**
   - `HYV3ForCausalLM` (`hy_v3`) GGUF load, `49`-layer offload, and actual VRAM margin on Colab L4 (`22,369 MiB` free observed vs `~20.8-21.2 GiB` projected at `CTX_SIZE=32768`, near `b12019` `-fit on`'s `1,024 MiB` free-memory target — inspect `llama.log` `common_params_fit_impl`; if `-fit on` drops any layers to CPU, lower `CTX_SIZE` to `16384`).
   - English adult H-scene refusal rate, `STYLE_ADULT` quality, and `s2twp` one-to-many character spot checks (`干`/`发`/`后`).
