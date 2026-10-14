# tools/ ??operational scripts that paid for themselves

These are the scripts that were actually used to run, verify and audit a batch.
They are deliberately small and parameterised: no hard-coded user paths, no
secrets. Each one exists because a mistake made it necessary ??see
`llm_handoff/14_thrashing_postmortem.md`.

| Tool | Run it | Why it exists |
|---|---|---|
| `pull_watch.py` | `python tools/pull_watch.py --instance <id> --tag <tag> [--destroy]` | **As soon as an instance exists.** Measures the REAL pull rate (manifest layer sizes vs completed layers) and gives up early on a host that keeps retrying or stops progressing. The advertised bandwidth filter is not evidence: a host was kept 35 minutes ($0.125) before its pull was called bad, because a 12 GB layer downloads for ~10 minutes with **no status change**, so "slow" looks exactly like "stuck". |
| `gate.py` | `python tools/gate.py --config config.json` | **Before any batch.** Proves the model can actually translate (3 lines in, exactly 3 non-empty Chinese lines out, no ` thinking`/template noise) and that the OCR detector loaded. Skipping this once meant starting a batch with a model that could not translate. |
| `verify_batch.py` | `python tools/verify_batch.py --dir <folder>` | **Before delivering.** Pairs each input archive with its `_MTL_zhTW` output **by name** (page counts collide) and checks page count, page order, image dimensions and translation coverage. |
| `audit_detection.py` | `python tools/audit_detection.py --dir <folder>` | **Before claiming a quality improvement.** Compares every *input* page with its *delivered* page by text-stroke density. If the delivered page looks the same as the source, nothing was erased there - that is how 74 of 1034 delivered pages were found to still contain untranslated text. Needs only numpy + opencv, no GPU, no money. |
| `run_batch.py` | `python tools/run_batch.py --config config.json --run 250 237` | Runs volumes sequentially on an already-rented machine, verifies each, and **destroys the machine only when everything verified**. Keeps the machine on failure (it is the debugging environment). |
| `audit_sheets.py` | `python tools/audit_sheets.py --dir <folder>` | Builds a contact sheet of the low-block pages **from the delivered archives** so a human can eyeball which pages still show the source language. **Do this before acting on the audit number** - it is how the "74 pages" over-call was discovered (they were almost all SFX, which this pipeline keeps on purpose). |
| `collect.py` | `python tools/collect.py --dir . --dest ComicsZHTW` | Collects the delivered archives into one folder, **verifying page count and order before moving each one**, and writes `MANIFEST.txt`. Nothing unverified can end up in the delivery folder. |

## Typical sequence

```powershell
# 0) free, local
python hanhua.py selftest
python hanhua.py auto -i sample.zip --ocr mock
python deploy/test_detection.py                 # detection geometry + adaptive flow

# 1) confirm the image tag exists (never rent for an unpublished tag)
#    https://hub.docker.com/v2/repositories/<repo>/tags/<tag>   -> 200

# 2) rent ONE machine, PROVE its pull is fast enough, then gate it
python hanhua.py vast up --yes --image <repo>:<tag> --baked --max-price 0.20 --write-config config.json
#    then, immediately: measure the pull and drop a bad host early
#    python tools/pull_watch.py --instance <id> --tag <tag> --destroy
python tools/gate.py --config config.json

# 3) run the volumes (verifies each, tears down on success)
python tools/run_batch.py --config config.json --run 250 237

# 4) audit and deliver
python tools/verify_batch.py --dir <folder>
python tools/audit_detection.py --dir <folder>
python tools/audit_sheets.py --dir <folder>     # optional, for human review
```

## Notes

- `gate.py`, `verify_batch.py` and `audit_detection.py` are safe to run at any
  time: they only read files and make HTTP requests to the machine you point
  them at.
- `run_batch.py` destroys a machine. It only destroys the instance recorded in
  `vast_current.txt`, and only after every volume verified.
- Nothing here rents a machine. Renting happens through exactly one guarded
  path: `hanhua.py vast up` (see `deploy/VASTAI_PLAYBOOK.md` R7).
