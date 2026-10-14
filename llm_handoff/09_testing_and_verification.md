# 09 — Testing and verification

Everything here is either free or already paid for. Nothing in this document
requires renting a machine, and **renting is not allowed without asking** (see
`deploy/VASTAI_PLAYBOOK.md` R1.1).

## Free, local (run these after any code change)

```bash
python hanhua.py selftest                 # synthetic page, mock provider, no network
python hanhua.py auto -i book.cbz --ocr mock   # real archive, mock OCR, no GPU/API cost
python deploy/test_detection.py           # unit tests: rotation geometry, dedup, adaptive flow
python -m py_compile $(git ls-files '*.py')    # every module compiles
bash -n deploy/container_start.sh deploy/net_check.sh
```

`deploy/test_detection.py` is the only test of the detection code, and it is
**torch-free**: it stubs `fastapi`, imports the real `ocr_server`, and checks the
rot90/rot270 box mappings against numpy ground truth, the IoU dedup, and the
adaptive escalation flow with a fake detector. It exists because the geometry is
easy to get subtly wrong and impossible to eyeball.

## The pre-flight gate (before any batch, on the machine)

```bash
python tools/gate.py --config config.json
```

This replaced a bare `ping`, which was not enough. It checks:

1. the OCR service is healthy **and its neural detector actually loaded**
   (`detector: true` - it silently fell back to EasyOCR for a whole project);
2. the LLM answers;
3. **three Japanese lines come back as exactly three non-empty Chinese lines**
   with no ` thinking` / `<|im_start|>` / template noise.

Point 3 is the one that matters: a wrong chat template, or a reasoning model
emitting its scratchpad into the answer, would ruin every volume while every
other check still passed.

## Delivery verification (required, and now a tool)

```bash
python tools/verify_batch.py --dir .        # pairs input->output BY NAME
```

Checks per volume: identical page count, **identical page names in the same
order**, unchanged image dimensions, and translation coverage (every block with
source text has a non-empty translation). Page count alone is not an identity -
two volumes in this project had 237 pages.

Then:

```bash
python tools/audit_detection.py --dir .     # did any page keep its source text?
python tools/audit_sheets.py  --dir .       # contact sheets, for a human to glance at
python tools/collect.py --dir . --dest ComicsZHTW
```

`audit_detection.py` compares the text-stroke density of each **input** page with
its **delivered** page. **Read its caveat first:** the metric cannot tell dialogue
from onomatopoeia/SFX, and this pipeline keeps SFX on purpose (`keep_sfx: true`).
It is a triage list, not a defect count - build the contact sheets and have a
human look before changing any code. (A first run called 74 of 1034 pages
defective; they were almost all SFX, and a "fix" was built and measured against
that number before anyone looked. See `14_thrashing_postmortem.md`.)

## Before renting, and immediately after

```bash
python tools/rent_and_run.py --yes --image <repo>:<tag> --tag <tag> --run <pages>...
```

That is the whole flow: rent one machine → **measure its pull**
(`tools/pull_watch.py`; a bad host is destroyed and blacklisted) → gate → run the
volumes → verify each → tear the machine down. It refuses to rent without `--yes`
and refuses to rent while an instance is already recorded.

## Diagnostics when something is off (all read-only)

| Tool | Question it answers |
|---|---|
| `tools/stream_check.py` | does this endpoint drop non-streaming requests? (the experiment that found the root cause: 4/6 vs 6/6) |
| `tools/llm_check.py` | is the translation endpoint flaky right now? |
| `tools/prep_time.py` | how long does `/prepare` take on real pages? (a page slower than the host's idle timeout gets dropped) |
| `tools/ssh_check.py` | processes, GPU, memory, `/root/ocr.log`, `start.log`, OOM messages on a live machine |
| `tools/quant_eval.py` | what would a stronger quantisation cost? (real GGUF sizes + live vast.ai prices) |
| `tools/local_probe.py` | the vendored detector's non-model paths on a real page, without torch or a GPU |

## What the build-time smoke test covers

`deploy/Dockerfile` ends with a step that **blocks every outbound socket** and
then:

1. `GET /health` and asserts `detector is True`;
2. runs `detect()` and asserts it returns a text mask;
3. runs `inpaint()` (LaMa) and `get_mocr()` (manga-ocr);
4. calls `POST /prepare` end to end.

So a missing model cache, a model that still wants the network, or a broken
dependency **fails the build** instead of a paid GPU box. That is the
machine-checked "the container runs offline" guarantee.

## Regression checklist after code changes

1. `python -m py_compile` on all modules; `bash -n` on shell scripts.
2. `deploy/test_detection.py` (if the detection code changed).
3. `hanhua.py selftest` and `hanhua.py auto --ocr mock` on a small archive.
4. `tools/verify_batch.py` on the result (page count, order, dimensions, coverage).
5. If the Dockerfile or `ocr_server.py` changed: rebuild the image (the build
   smoke test covers the service) - but **only tag it if a build is actually
   wanted**; a plain master commit does not trigger CI.
