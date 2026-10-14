# 07 — Code review

> **Historical.** This is the review as performed early in the project (against
> `v0.1.0`). Every finding below was fixed then. The code has changed a lot
> since: the detector is now **vendored**, the LLM client **streams**, detection
> is **adaptive**, and the image is at **`v0.5.0`**. For the current state read
> `10_current_state_and_next_steps.md`; for the current tools and tests read
> `09_testing_and_verification.md`.

A strict review of the Dockerfile, the vast.ai controller, the OCR service, and
the pipeline modules. Findings marked 🔴 caused money or data loss.

## Fixed findings

| ID | Severity | Location | Defect | Fix |
|---|---|---|---|---|
| V1 | 🔴 | `vast.up()` | pre-run cleanup destroyed **all** instances, including the user's others | only destroy instances labelled by this tool |
| V2 | 🔴 | `vast.up()` | no exception handling → a leaked machine kept billing | wrap the loop body; destroy on any unexpected error |
| K1 | 🔴 | `vast._generate_key()` | Windows key ACL too open → ssh ignored the key (the biggest money sink) | run `icacls` after generating |
| K2 | 🟠 | `vast._generate_key()` | subprocess output decoded as UTF-8 crashed a reader thread | pass `encoding/errors` |
| O1 | 🟠 | `ocr_server.build_mask()` | without a segmentation mask the mask was the whole box → the model redrew borders/artwork | build a **text mask** (Otsu inside each box, dilated) |
| O5 | 🟠 | `ocr_server./prepare` | 500 responses had an empty body → undebuggable | return the traceback in the response detail |
| H2 | 🟠 | `hanhua.auto` | a failed run deleted the work dir (losing cleaned pages) | keep the work dir on failure; add `--keep-work` |
| W1 | 🟠 | `vast.wait_http()` | an empty model list counted as ready | require a **non-empty** model list |
| W2 | 🟠 | `vast.wait_http()` | no download-speed gate | read `hf_mbps` from `/health`; fail fast below threshold |
| B1 | 🟠 | `vast.bootstrap_script()` | the OCR service started after the model pull, so `/health` was unreachable during it | start OCR first, then pull |
| S1 | 🟠 | `vast.search_offers()` | sorting by reliability first picked expensive machines | cheapest inside a reliability band; lower default max price |
| N1 | 🟠 | `deploy/net_check.sh` | an unrelated site's 403 failed almost every machine | only require the sources actually used |
| L1 | 🟠 | `ocr_server.inpaint()` | the model's 8-pixel padding was not cropped back | crop to the original size |
| P1 | 🟡 | `hanhua.py` | a flag was defined twice (global and subcommand) | define once |

## Dockerfile review

Design: a CUDA+PyTorch base → system libs → pinned Python deps (with a
constraints file protecting CUDA PyTorch) → detector from its code host → ollama
→ pre-downloaded weights → **build-time smoke test** → container start script.

| Check | Verdict | Note |
|---|---|---|
| `libgl1` / `libglib2.0-0` | required | OpenCV import fails without them |
| `git` + detector from code host | required | not on the package index; falls back if it fails |
| constraints protecting torch/torchvision | important | prevents a CPU wheel replacing CUDA torch |
| `numpy<2` | important | ABI compatibility with older OpenCV/PyTorch |
| pinned `transformers` | required | newer releases need a newer torch than the base |
| multipart helper | required | upload endpoint otherwise 500s |
| pre-downloaded weights | good | removes a ~1 GB fetch per boot |
| **build-time smoke test** | core | moves dependency failures to the free build step |
| `.dockerignore` | good | tiny build context |
| optional model bake | optional | removes the boot-time download; image grows ~9 GB |

### Not fixed / accepted trade-offs

| ID | Severity | Issue | Recommendation |
|---|---|---|---|
| D1 | 🟡 | base image pinned by tag, not digest | pin by digest for reproducibility |
| D2 | 🟡 | the detector's weights may not be bundled → it falls back to a general detector (no segmentation) | download the weights explicitly, or assert the detector loads at build time |
| D3 | 🟡 | runs as root | conventional for this kind of container; add `USER` for sensitive environments |
| D4 | 🟡 | no `HEALTHCHECK` | external polling covers it |
| D5 | 🟡 | image is large (~8–10 GB) | registry cache / volume cache |
| D6 | 🟡 | model singletons are loaded lazily and not thread-safe | add a lock, or keep a single worker |

## Remaining design issues (not fixed; decide if they matter)

| ID | Location | Issue | Impact |
|---|---|---|---|
| R3 | `render._wrap` | Latin text may break mid-word | English dialogue readability |
| R4 | `render._fit_vertical` | vertical text is fitted to "fits", may end up small | layout aesthetics |
| T1 | `terms`/`normalize` | using a translation model for JSON tasks is unreliable | consistency enforcement is limited; use a general LLM for these |
| T2 | `_auto_glossary` | only keeps katakana/honorific names, misses kanji names | glossary may be incomplete |
| P2 | `ocr._http` | multipart only, no retry | one page failure aborts the run (mitigated by keeping the work dir) |

## Verification performed

- `py_compile` on all Python files: pass.
- `bash -n` on the container start script and deploy scripts: pass.
- `hanhua.py selftest` (mock, both profiles): pass.
- New flags (`vast up --image/--baked`, `auto --keep-work/--strict`) parse.
- The Docker image: **built and pushed successfully** on tag `v0.1.0`; the
  build-time smoke test passed in CI (13m7s run).
