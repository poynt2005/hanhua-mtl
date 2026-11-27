# 20 — Sakura Colab Workflow (`sakura-ollama-colab`)

> **Branch:** `sakura-ollama-colab` (branched from `origin/sakura-ollama` at `a94c8fb`)
> **Updated:** 2026-11-28 01:30 +08:00
> **Agent:** Antigravity
> **Model:** Gemini 4 Neon
> **Status:** READY — Milestone 12 complete (Option A exit-code separation `0/1/2`, mid-run `llama-server` crash guard, 5x Drive FUSE checkpoint write retry, `gpu.log` resilience, C10/C11 hardening; verified across 18/18 offline tests + isolated 14,269-row `rc=2` & `rc=0` dry-runs + 0 sanitization hits)
> **Session Postmortem & Full Context:** See [21_colab_session_postmortem.md](21_colab_session_postmortem.md) for the complete C1–C12 failure table, 10-item operator accountability log, F1–F11 review fixes, 9 permanent guardrails, and live Colab L4 specs.

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
Replace the `vast.ai` rental workflow for the Japanese -> Traditional Chinese Galgame script translation branch (`sakura-ollama`) with a zero-compile **Google Colab L4 (24 GB) + Google Drive (5 TB) one-click workflow**:
1. **Free CPU GGUF & Binary Deployer (`deploy_models_colab.ipynb`):** Runs on a free CPU runtime, mounts Drive, creates fixed folders, downloads pinned GGUF weights from HuggingFace (with `hf-mirror` fallback) plus the prebuilt `llama-server` tarball (`colab-l4-llama-b12019`), and verifies exact size and SHA-256 on both local SSD and Drive before atomic rename.
2. **Colab L4 One-Click Translator (`sakura_colab.ipynb` + `runner.py`):** Extracts the pinned prebuilt `llama-server-l4-b12019.tar.gz` bundle from Drive `bin/` (or GitHub Release fallback) with zero compilation, stages the 12.12 GB GGUF once to local NVMe SSD, enforces 4 hard GPU gates, runs 3-line pre-flight Gate + tok/s check, runs deduplicated 4-way parallel `Sakura-GalTransl-14B-v3.8` batch-8 translation with SHA-1 `unit_key` checkpoint/resume, deterministic quote/placeholder/kana/multiline repairs, `no_cjk`/`latin_only` language QA, `--repair-only` + `<job>.overrides.jsonl` support, writes `.DONE.json`, flushes Drive, and protects GPU quota with `DISCONNECT_ON_ERROR=False` + `IDLE_UNASSIGN_MINUTES=15`.
3. **Local Option A Drive Bridge (`drive_bridge.py` + `skill/SKILL.md`):** Pure-stdlib Windows CLI that checks `GoogleDriveFS.exe`, auto-detects the single mounted Google Drive containing `galgame_mtl` (supporting multi-account Drive e.g. `H:\`), validates strict `.jsonl` + `glossary.json`, syncs `schema.py`/`runner.py` to Drive `scripts/sakura/`, atomically pushes `<job>.jsonl` + `<job>.manifest.json`, checks status, and pulls verified outputs after `.DONE.json` SHA-256 verification.
4. **Agent Migration SOP (`AGENT_SOP_COLAB.md`):** Explicit migration guide for agents arriving with `vast.ai` handoff context.

---

## 2. Locked Constants, Architecture & Quality Contracts

### 2.1 Runtime & Model Constants
| Item | Value |
|---|---|
| Colab Drive root | `/content/drive/MyDrive/galgame_mtl` |
| Model | `SakuraLLM/Sakura-GalTransl-14B-v3.8` (`Sakura-Galtransl-14B-v3.8.gguf`, Q6_K) |
| Pinned HF rev | `10bab61b45d8970dc5c5fd7a4aec920d49a3cd8e` |
| Exact model size & SHA-256 | `12,124,681,536` bytes / `6e32fcf5bfdf78114745d2bf6bd87fb107ad7f68a611f4c2a886a4378846e334` |
| Prebuilt server bundle | GitHub Release `poynt2005/hanhua-mtl` tag `colab-l4-llama-b12019`, `llama-server-l4-b12019.tar.gz`, `773,245,209` bytes, SHA-256 `802a169e33dde5e73403c2a36f3d6dbfd2d34d5e9ccd898e79a1eb899bcac947`, `43` entries |
| Server flags | `127.0.0.1:11434`, ctx `16384`, `-np 4`, `-ngl 999`, `--device CUDA0 --cont-batching -lv 4`, `LLAMA_ARG_UI=0` (never pass redundant `LLAMA_ARG_LOG_VERBOSITY`) |
| GPU Gates | Gate 0: total VRAM >= `22,000 MiB` (rejects T4 before staging); Gate 1: `--list-devices` lists `CUDA0`; Gate 2: `-lv 4` `llama.log` shows `offloaded X/X layers to GPU` (polled up to `10 x 0.5s`); Gate 3: used VRAM >= `12,000 MiB`; Gate 4: 3-line pre-flight throughput >= `8.0 tok/s` |
| Sampling | Batch: `temperature=0.3` (retry `0.1`); Single-unit: `(0.3, True)` -> `(0.1, True)` -> `(0.0, False)` (`top_p=0.8`, `repeat_penalty=1.0`, `max_tokens=2048` batch / `512` single, `stream=True`) |
| Traditional conversion | Default OpenCC `s2t` (per operator historical rule; `s2twp` selectable via `--tw-mode`) |

### 2.2 Stable Unit Identity (`unit_key`) & Checkpoint Migration
- `make_unit_key(kind, speaker, text)` computes `sha1(f"{kind}\0{speaker or ''}\0{text}".encode("utf-8"))`.
- Every checkpoint row written by the current runner records `unit_key`, `wid` (display/order label only), `zh_cn`, plus optional `raw` and `flags`.
- `_load_checkpoint(ckpt_path, legacy_wid_to_key=...)` migrates legacy `wid`-only checkpoint files cleanly using the current input's first-pass `w000001..` -> `unit_key` mapping. Adding or editing `override` entries shifts subsequent `wid`s safely without invalidating completed cache entries.

### 2.3 Prompt Fidelity & Single-Unit Targeted Retries
- **Strict user trigger:** `build_sakura_messages` always emits exact official GalTransl v3.8 syntax (`历史翻译：` / `参考以下术语表...：` / `根据以上术语表的对应关系和备注，结合历史剧情和上下文，将下面的文本从日文翻译成简体中文：\n` + source lines). Nothing may ever be injected between `：` and `\n` (enforced by HTTP 400 in `test_all.py`).
- **System-only retry hints (`_hint_for_errors`):** Single-unit retries append targeted constraints (`placeholder_mismatch`, `quote_asymmetry_mismatch`, `residual_kana`, `no_cjk`/`latin_only`) only to `SYSTEM_PROMPT`, never to the user prompt trigger.
- **`ui` / `system` isolation:** History is forced empty (`[]`) when `kind in ("ui", "system")`.

### 2.4 Deterministic Normalization & Repairs
1. **Placeholders (`encode_placeholders` / `decode_placeholders`):** Normalizes `{{TAG_n}}` / `⟦n⟧` / `\Cmd[n]` to `⟦0⟧`, `⟦1⟧`, ... on prompt construction and injects `⟦i⟧->⟦i⟧ #占位符，原样保留` into `term_lines`; decodes spaced variants `⟦ 0 ⟧` (`flag: repair_placeholder`) and restores original tokens before validation.
2. **Speaker stripping:** Strips outer and inner mirrored `speaker:` prefixes longest-first (`strip_speaker_prefix`, `strip_inner_speaker`).
3. **Quote repair (`repair_quotes`):** Deterministically repairs 8 structural cases without altering inner wording:
   - Half-width corner quotes `｢｣` -> `「」` (`repair_halfwidth_quotes`)
   - Paired curly quotes `“”` -> `「」` (`repair_curly_quotes`)
   - Paired ASCII `"` -> `「」` (`repair_ascii_quotes`)
   - Paired `『』` -> `「」` when source uses `「」` without `『』` (`repair_double_brackets`)
   - Chinese title marks `《》` (`repair_title_quotes`) and lenticular brackets `【】` (`repair_lenticular_quotes`) when source uses `「」` without those marks
   - Duplicated outer quotes `「「...」」` (`trim_dup_open_quote`, `trim_dup_close_quote`, `trim_outer_quote_pair`)
   - Asymmetric split-quote trims (`「...` + `「...」` -> `trim_close_quote`; `...」` + `「...」` -> `trim_open_quote`)
   - Missing outer single/pair corner quotes on full-wrapped dialogues (`add_quote_pair`, `add_close_quote`, `add_open_quote`)
4. **Heart glyphs & full-width indents:** Exclusive `♥`<->`♡` normalization (`heart_norm`); exact multi-`\u3000` leading indent count preserved (`lead_fw = len(src) - len(src.lstrip("\u3000"))`) so multi-indent RPG descriptions stay intact and `normalize_output_line` is idempotent.
5. **Kana boundaries & glossary substitution:**
   - `_JA_PRESENT_RE = r"[\u3041-\u3096\u30a1-\u30fa\u4e00-\u9fff]"`
   - `_KANA_RE = r"[\u3041-\u3096\u309d-\u309f\u30a1-\u30fa\u30fd-\u30ff]"` — catches all real Hiragana/Katakana letters while **excluding** katakana middle dot `・` (`U+30FB`), prolonged sound mark `ー` (`U+30FC`), and `゠` (`U+30A0`).
   - When `_KANA_RE` matches and a matched glossary entry (`_KANA_RE` in `src`) appears verbatim in the output, `glossary_subst` replaces `src` -> `dst` before retry/validation.
6. **Multiline units (`\n`):** `_translate_single_multiline` splits `unit["text"]` on `\n`, translates each non-empty segment individually through `_translate_single_unit` (speaker attached to first non-empty segment only, segment history maintained), verifies exact segment count, rejoins with `\n`, and runs full-unit validation.
7. **Language QA & `.DONE.json` Gate (`check_language_qa`, `is_allowlisted_no_cjk`):**
   - When source has Japanese (`_JA_PRESENT_RE`), output must contain at least one CJK character (`_CJK_RE`), unless `is_allowlisted_no_cjk` accepts it (non-empty stripped core that is pure numbers/symbols OR ASCII word tokens appearing verbatim case-insensitively in `src_text`, allowing verbatim ASCII command-example lines while rejecting English translations and truncated punctuation-only outputs).
   - `qa_report.json` tracks `manual_overrides`, `repaired`, `missing`, `failures`, `no_cjk`, `latin_only`, `tw_mode`, `roundtrip_mismatches`, and `watch_chars`.
   - `.DONE.json` is written (and exit code `EXIT_OK = 0` returned) **only** when `missing == 0 and failures == 0 and no_cjk == 0 and latin_only == 0`.
8. **Three-Tier Exit Codes (Option A), FUSE Resilience, Actionable Failures & Notebook Lifecycle:**
   - **`EXIT_OK = 0`**: All jobs passed 100%, `.DONE.json` written -> `[DONE]` -> `AUTO_DISCONNECT` unassigns L4.
   - **`EXIT_PARTIAL = 2`**: All jobs completed and saved `zh_CN.jsonl`, `zh_TW.jsonl`, `qa_report.json`, `checkpoint.jsonl`, and `<job>.failures.jsonl` while `llama-server` stayed healthy (`proc.poll() is None`), with remaining `failures`/`missing`/`no_cjk`/`latin_only` -> `.DONE.json` NOT written -> notebook prints calm `[PARTIAL]` summary (no `RuntimeError`), flushes Drive, and honors `AUTO_DISCONNECT=True` immediately (zero idle GPU burn).
   - **`EXIT_FATAL = 1`**: Raised/returned ONLY on startup Gates 0-4, invalid inputs/manifest, persistent Drive FUSE write errors, or unexpected mid-run `llama-server` exit (`proc.poll() is not None`, checked immediately after `translate_job`). `DISCONNECT_ON_ERROR=False` retains L4 up to `IDLE_UNASSIGN_MINUTES=15` with `STOP THE RUNTIME NOW` banner.
   - **Drive FUSE resilience:** `_append_checkpoint_rows` retries transient `OSError` up to `5` times (`0.25s` exponential backoff) before updating in-memory `done_map` / `progress_counter`; `_log_gpu_progress` isolates `gpu.log` `OSError` as a non-fatal warning.
   - `<job>.failures.jsonl` emits `{unit_key, wid, text, errors, attempts, expected_placeholders, observed_placeholders, suggested_fix, needs_override, source_kind}`.
   - `<job>.overrides.jsonl` (auto-discovered in `sakura/input/` or via `--overrides`) accepts `translation` / `zh_cn` / `override` keyed by `id`, `unit_key`, or `kind` + `speaker` + `text`, passing directly through `to_simplified` without GPU usage.
   - `sakura_colab.ipynb` defaults: `AUTO_DISCONNECT=True`, `DISCONNECT_ON_ERROR=False`, `IDLE_UNASSIGN_MINUTES=15`, `REPAIR_ONLY=False`, `TARGET_JOB=""`.

---

## 3. Progress Log

| Timestamp (+08:00) | Milestone | Outcome | Notes |
|---|---|---|---|
| 2026-11-26 01:25 | Milestone 1: Branch & Core Implementation | Done | Created `deploy/colab-sakura/` (`schema.py`, `runner.py`, `drive_bridge.py`, `deploy_models_colab.ipynb`, `sakura_colab.ipynb`, `AGENT_SOP_COLAB.md`, `skill/SKILL.md`, `README.md`, `test_all.py`) and updated `llm_handoff/INDEX.md` |
| 2026-11-26 01:28 | Milestone 2: Code Review, Offline Tests & Sanitization | Done | Fixed 5 review items (closed `llama.log` parent fd, simplified fallback `build/bin` search, removed redundant ternary, added Drive glossary warning on `push` without `--glossary`, wrapped `drive.flush_and_unmount()` in `try/except` inside notebooks so `runtime.unassign()` always executes). `py_compile` + `test_all.py` (5/5 tests) passed; sanitization scan 0 hits |
| 2026-11-26 11:51 | Milestone 3: Live Colab Failure Fix | Done | First real Colab run exited rc=1 silently with empty `bin/` and `work/` (notebook `subprocess.run` hid child stdout/stderr and saved no `runner.log`, while `b12019` CMake had `LLAMA_OPENSSL=ON` + deprecated `LLAMA_CURL`). Fixed: live-streamed `Popen` + tee to `sakura/work/runner.log` + tail in exception; explicit `_find_nvcc()` + `-DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF`; Gate 0 VRAM check moved before build/staging; `stage_model` copies once to NVMe then verifies SHA-256; replaced `copy2` with `copyfile` and ignored `OSError` on `fsync` for Drive FUSE; ignored OS metadata files in `input/`. `py_compile` + `test_all.py` (5/5) + sanitization (0 hits) passed |
| 2026-11-26 12:01 | Milestone 4: Fix `b12019` `llama-ui-assets` CMake Failure | Done | Live Colab `runner.log` showed `cmake --build` failed at `llama-ui-assets` (`ui_validate_assets`: shallow `--depth 1` clone produced `b1`, fell back to `ggml-org/llama-ui/resolve/latest/dist.tar.gz`, mismatching `b12019` `bundle[hash].js` check). Verified in `b12019` `scripts/ui-assets.cmake` and added `-DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF` (`n_assets == 0` early return). Re-ran `py_compile`, `test_all.py` (5/5), sanitization (0 hits) |
| 2026-11-26 12:20 | Milestone 5: Prebuilt-First Zero-Compile `llama-server` & Free-CPU Pre-Caching | Done | Eliminated slow paid-L4 compilation by switching `ensure_llama_server` to prebuilt bundles & GitHub Release caching. Added `DEPLOY_LLAMA_BIN = True` free-CPU pre-caching to `deploy_models_colab.ipynb`. Re-ran `py_compile`, `test_all.py` (6/6), sanitization (0 hits) |
| 2026-11-26 12:35 | Milestone 6: Complete Removal of Colab Build Code & Pinned Release Bundle | Done | Repacked official `ggml-org/llama.cpp` `b12019` Ubuntu x64 CUDA 12.8 `llama-server` + runtime `.so*` libs (`773,245,209` bytes, SHA-256 `802a169e33dde5e73403c2a36f3d6dbfd2d34d5e9ccd898e79a1eb899bcac947`, 43 entries) and published to GitHub Release `colab-l4-llama-b12019`. Removed ALL `git clone`/`cmake`/`--rebuild-bin`/`REBUILD_LLAMA_BIN` code from `runner.py` and `sakura_colab.ipynb`; updated `deploy_models_colab.ipynb` to deploy the pinned tarball directly to Drive `bin/` with dual SHA-256 verification. Re-ran `py_compile`, `test_all.py` (6/6), sanitization (0 hits) |
| 2026-11-26 13:21 | Milestone 7: Fix `b12019` Log Verbosity (`-lv 4`) for Gate 2 GPU Offload Verification | Done | Live Colab Run #3 passed Gate 0, Drive `bin/`, 12.12 GB SSD staging, Gate 1 (`CUDA0`), and 4.6 s model load, then failed Gate 2 because `b12019` `common/log.cpp` requires `-lv 4` to emit `offloaded X/Y layers to GPU`. Added `-lv 4` + clean `proc.terminate()` + `llama.log` tail on Gate 2 errors + temp retry in `run_preflight_gate`. Re-ran `py_compile`, `test_all.py` (6/6), sanitization (0 hits) |
| 2026-11-26 13:35 | Milestone 8: Complete Code Review Fixes (F1, F2, F4, F5, F6, F7) | Done | Polled `llama.log` up to `10 x 0.5s` for async logger flush race (F1); fixed `roundtrip_mismatches` vs `cn` (F2); added `DISCONNECT_ON_ERROR` toggle (F4); switched `drive_bridge.py` to `copyfile` + `Prebuilt bin` check in `doctor` (F5); cleaned stale comment (F6); moved GPU logging under `gpu_log_lock` (F7) |
| 2026-11-26 14:30 | Milestone 9: Fix Live Run #4 `strict_line_hint` Prompt Echo & Split-Quote/Default Fixes | Done | Run #4 passed Gates 0-3 (`49/49` layers, `14378 MiB`, `20.84 tok/s`) then failed Gate 4 sample 1 due to `strict_line_hint` injected after `：`. Removed `strict_line_hint` completely, set `(0.3, 0.1, 0.0)` gate fallback, removed redundant `LLAMA_ARG_LOG_VERBOSITY`, added asymmetric split-quote & heart-glyph normalization, defaulted `DISCONNECT_ON_ERROR=False`, hardened `test_all.py` mock handler to HTTP 400 on prompt drift |
| 2026-11-28 00:50 | Milestone 10: Stable `unit_key`, Quote/Placeholder/Multiline Repair, Language QA & `--repair-only` (C10) | Done | Live Run #5 completed the 1,808-row validation job (`24.2 tok/s`, 92-100% L4 util); a subsequent 14,269-row (`2,235` unique-unit) production job finished `2,215` units (`24.1 tok/s`) with `20` residual failures (`46` missing rows) + `12` undetected `no_cjk` / English outputs (`5` verbatim command-example lines + `7` untranslated/truncated lines). Implemented SHA-1 `unit_key` + legacy `wid` migration, generalized `repair_quotes`, internal `⟦0⟧` placeholder encode/decode, `overrides.jsonl` + kana glossary substitution, `\n` multiline segment split/rejoin, `no_cjk`/`latin_only` QA + allowlist, `--repair-only`/`--job`/`--overrides`, actionable `failures.jsonl`, and `IDLE_UNASSIGN_MINUTES=15` + `STOP THE RUNTIME NOW` safeguard. Added 9 test suites (`15/15` passing), verified isolated 14,269-row repair dry-run (`cached=2215 | todo=20`), 0 sanitization hits |
| 2026-11-28 01:00 | Milestone 11: Real-Data Regex & Validator False-Positive Fixes (C11) | Done | Auditing all validators/regexes against all 14,269 real translated rows revealed: (1) `_KANA_RE` (`\u30a0-\u30ff`) falsely flagged katakana middle dot `・` (`U+30FB`) and prolonged mark `ー` (`U+30FC`) as `residual_kana` in 9 valid rows; (2) `normalize_output_line` collapsed multiple leading `\u3000` indents to 1 across 60 multi-indent rows; (3) `repair_quotes` missed `《》` / `【】` title-quote repairs. Narrowed `_KANA_RE` / `_JA_PRESENT_RE` to real kana letters, preserved exact leading `\u3000` count (verified idempotent), added `《》` / `【】` quote repair, and confirmed 0 errors & 0 non-indent unintended mutations across all 14,269 real rows |
| 2026-11-28 01:30 | Milestone 12: Option A Exit-Code Separation (`0/1/2`) & FUSE I/O Resilience (C12) | Done | Separated `EXIT_OK = 0`, `EXIT_FATAL = 1`, `EXIT_PARTIAL = 2` so completed runs with minor residual failures save all outputs, skip `.DONE.json`, print calm `[PARTIAL]` summary in `sakura_colab.ipynb` (no red `RuntimeError`), and still honor `AUTO_DISCONNECT=True`; added `proc.poll() is not None` mid-run server crash guard, 5-attempt exponential-backoff `_append_checkpoint_rows` (committing `done_map` post-write), and `gpu.log` `OSError` warning isolation. Verified 18/18 tests + isolated 14,269-row `rc=2` & `rc=0` dry-runs + 0 sanitization hits |

---

## 4. Verification Status (Honest State)

| Check | State | Notes |
|---|---|---|
| `py_compile` on `schema.py`, `runner.py`, `drive_bridge.py`, `test_all.py` | `done` (pass) | Executed after Milestone 12 edit |
| Offline `test_all.py` (18 suites covering schema, normalization, 8 quote-repair fixtures, `・`/`ー` non-kana & multi-`\u3000` indent regression, kana/glossary/overrides, placeholder normalization + failure record, multiline segment split/rejoin, `no_cjk`/`latin_only` + allowlist, stable `unit_key` + legacy migration, `--repair-only` `cached=2215 \| todo=20`, `EXIT_PARTIAL=2` + DONE gate block, Drive FUSE checkpoint retry + `gpu.log` `OSError` resilience, mid-run server crash `EXIT_FATAL`, ID parity, prebuilt bundle, `.ipynb` AST + `[PARTIAL]` + disconnect matrix + strict HTTP 400 prompt check) | `done` (18/18 pass) | Executed after Milestone 12 edit using local mock SSE & HTTP servers and temp dirs |
| 14,269-row real repaired output audit (`validate_unit_translation` + `normalize_output_line` idempotency) | `done` (pass) | `errs=0`, `non_indent_muts=0`, `idempotent_muts=0` across all 14,269 real production rows (Milestone 11) |
| Isolated 14,269-row repair dry-run (`--repair-only` on read-only temp copy of legacy 2,215-unit checkpoint) | `done` (pass) | Executed after Milestone 12 edit: verified Case 1 (`rc=2` partial, no `.DONE.json`, 14,269 rows written) and Case 2 with overrides (`rc=0`, `.DONE.json` written, stale `failures.jsonl` removed) |
| Sanitization scan & Drive sync verification (`drive_bridge.py doctor`, standalone export, SHA-256 parity) | `done` (0 hits) | Verified zero secrets, local paths, game names/prose, or Docker CI triggers |
| Live Colab L4 Gates 0-4 & 1,808-row run (Run #5) | `done` (pass) | `NVIDIA L4 23034 MiB`, `49/49` layers, `14378 MiB`, `24.2 tok/s`, 92-100% GPU util |
| Live Colab L4 `--repair-only` / post-C12 full production run | `not tested` | Requires operator live Colab L4 run after Milestone 12 push |

---

## 5. Operational Runbook & Incremental Repair Playbook

### Standard New Job
1. `python deploy/colab-sakura/drive_bridge.py doctor` (or `python <instructions_dir>/drive_bridge.py doctor` when working outside the repo)
2. `python deploy/colab-sakura/drive_bridge.py validate --input <job.jsonl> [--glossary <glossary.json>]`
3. `python deploy/colab-sakura/drive_bridge.py push --input <job.jsonl> [--glossary <glossary.json>] --name <job>`
4. Operator opens latest `sakura_colab.ipynb` on **Colab L4 GPU** (`DISCONNECT_ON_ERROR=False`, `IDLE_UNASSIGN_MINUTES=15`, `AUTO_DISCONNECT=True`) and runs `Ctrl+F9`.
5. `python deploy/colab-sakura/drive_bridge.py status --name <job>`
6. `python deploy/colab-sakura/drive_bridge.py pull --name <job> --dest <out_dir>`
7. **Post-pull Drive cleanup:** ONLY after `pull` passes SHA-256 + ID parity and local outputs are verified saved on disk, delete the completed job's residual files from `sakura/input/`, `sakura/work/`, and `sakura/output/` so the next Colab job never scans stale files. **Never** delete `models/`, `bin/`, or `scripts/`.

> **Standalone instructions export:** Every export to `<instructions_dir>/sakura-ollama-colab/` must bundle all **11 files** (the 9 files under `deploy/colab-sakura/` plus `20_sakura_colab_workflow.md` and `21_colab_session_postmortem.md`).

### Incremental Repair After Partial Failure (`failures > 0` or `no_cjk > 0`)
1. Inspect `sakura/work/<job>.failures.jsonl` (contains `unit_key`, `errors`, `expected_placeholders`, `observed_placeholders`, `suggested_fix`) and/or `sakura/output/<job>.qa_report.json`.
2. For proper-noun loanwords, missing tags, or remaining `no_cjk` legacy lines, prepare `<job>.overrides.jsonl` in `sakura/input/` (keyed by `id`, `unit_key`, or `kind` + `speaker` + `text` with `"translation": "..."`).
3. Re-run **without retranslating cached units**:
   - In Colab: set `REPAIR_ONLY = True` and `TARGET_JOB = "<job>"` in `sakura_colab.ipynb` -> `Ctrl+F9`; or
   - CLI: `python runner.py --drive-root <root> --api-base <url> --repair-only --job <job> [--overrides <path>]`
4. Verify `missing == 0 and failures == 0 and no_cjk == 0 and latin_only == 0` and `<job>.DONE.json` SHA-256 verification passes before `pull`.
