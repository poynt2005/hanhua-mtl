# tools/ ??operational scripts that paid for themselves

These are the scripts that were actually used to run, verify and audit a batch.
They are deliberately small and parameterised: no hard-coded user paths, no
secrets. Each one exists because a mistake made it necessary ??see
`llm_handoff/14_thrashing_postmortem.md`.

| Tool | Run it | Why it exists |
|---|---|---|
| **`rent_and_run.py`** | `python tools/rent_and_run.py --yes --image <repo>:<tag> --tag <tag> --run <pages>...` | **The whole flow in one command** (recommended entry point): rent one machine → measure its pull with `pull_watch.py` → gate → run the volumes → verify → tear down. It is the glue that used to exist only in the author's scratch directory, which meant a fresh clone could not run the flow unattended. |
| `pull_watch.py` | `python tools/pull_watch.py --instance <id> --tag <tag> [--destroy]` | **As soon as an instance exists.** Measures the REAL pull rate (manifest layer sizes vs completed layers) and gives up early on a host that keeps retrying or whose projected ETA is too long. The advertised bandwidth filter is not evidence: a host was kept 35 minutes ($0.125) before its pull was called bad, because a 12 GB layer downloads for ~10 minutes with **no status change**, so "slow" looks exactly like "stuck". |
| `gate.py` | `python tools/gate.py --config config.json` | **Before any batch.** Proves the model can actually translate (3 lines in, exactly 3 non-empty Chinese lines out, no ` thinking`/template noise) and that the OCR detector loaded. Skipping this once meant starting a batch with a model that could not translate. |
| `verify_batch.py` | `python tools/verify_batch.py --dir <folder>` | **Before delivering.** Pairs each input archive with its `_MTL_zhTW` output **by name** (page counts collide) and checks page count, page order, image dimensions and translation coverage. |
| `audit_detection.py` | `python tools/audit_detection.py --dir <folder>` | **Before claiming a quality improvement.** Compares every *input* page with its *delivered* page by text-stroke density. If the delivered page looks the same as the source, nothing was erased there - that is how 74 of 1034 delivered pages were found to still contain untranslated text. Needs only numpy + opencv, no GPU, no money. |
| `run_batch.py` | `python tools/run_batch.py --config config.json --run 250 237` | Runs volumes sequentially on an already-rented machine, verifies each, and **destroys the machine only when everything verified**. Keeps the machine on failure (it is the debugging environment). |
| `audit_sheets.py` | `python tools/audit_sheets.py --dir <folder>` | Builds a contact sheet of the low-block pages **from the delivered archives** so a human can eyeball which pages still show the source language. **Do this before acting on the audit number** - it is how the "74 pages" over-call was discovered (they were almost all SFX, which this pipeline keeps on purpose). |
| `collect.py` | `python tools/collect.py --dir . --dest ComicsZHTW` | Collects the delivered archives into one folder, **verifying page count and order before moving each one**, and writes `MANIFEST.txt`. Nothing unverified can end up in the delivery folder. |

### Diagnostics (read-only; use them when something is off)

| Tool | Run it | What it answers |
|---|---|---|
| `stream_check.py` | `python tools/stream_check.py --config config.json` | **Does this endpoint drop non-streaming requests?** This is the experiment that found the root cause of every aborted batch (4/6 non-streaming vs 6/6 streaming). |
| `llm_check.py` | `python tools/llm_check.py --config config.json` | Is the translation endpoint flaky right now? Hammers it with realistic batches and reports failures. |
| `prep_time.py` | `python tools/prep_time.py --config config.json --dir . --pages 250` | How long does `/prepare` take on real pages? (A page slower than a host's idle timeout gets dropped.) |
| `ssh_check.py` | `python tools/ssh_check.py --instance <id>` | Everything about a live machine over SSH: processes, GPU, memory, `/root/ocr.log`, `start.log`, OOM messages. |
| `quant_eval.py` | `python tools/quant_eval.py --repo <hf-repo>` | **What would a stronger quantisation cost?** Real GGUF sizes from HuggingFace + live vast.ai prices, and the cost per volume. This is how the Q6_K decision was made. |
| `local_probe.py` | `python tools/local_probe.py --vendor <checkout> --dir . --pages 250` | Exercises the vendored detector's non-model code paths (cv2 on the BGR view, `letterbox`, `refine_mask`, `build_mask`) on a real page, **without torch and without a GPU**. |

## Typical sequence

```powershell
# 0) free, local
python hanhua.py selftest
python hanhua.py auto -i sample.zip --ocr mock
python deploy/test_detection.py                 # detection geometry + adaptive flow

# 1) confirm the image tag exists (never rent for an unpublished tag)
#    https://hub.docker.com/v2/repositories/<repo>/tags/<tag>   -> 200

# 2) the whole remote flow, in one command:
#    rent one machine -> measure its pull -> gate -> run -> verify -> tear down
python tools/rent_and_run.py --yes --image <repo>:<tag> --tag <tag> `
       --config config.json --dir . --run 250 237

# 3) audit and deliver (also done automatically by run_batch, but re-runnable)
python tools/verify_batch.py --dir <folder>
python tools/audit_detection.py --dir <folder>
python tools/audit_sheets.py --dir <folder>     # optional, for human review
python tools/collect.py --dir <folder> --dest ComicsZHTW
```

`rent_and_run.py` refuses to rent without `--yes`, and refuses to rent while an
instance is already recorded in `vast_current.txt` (one machine at a time).

## Notes

- `gate.py`, `verify_batch.py` and `audit_detection.py` are safe to run at any
  time: they only read files and make HTTP requests to the machine you point
  them at.
- `run_batch.py` destroys a machine. It only destroys the instance recorded in
  `vast_current.txt`, and only after every volume verified.
- Nothing here rents a machine. Renting happens through exactly one guarded
  path: `hanhua.py vast up` (see `deploy/VASTAI_PLAYBOOK.md` R7).
