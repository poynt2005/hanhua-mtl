---
name: hymt2-colab-translate
description: Use when translating English adult Galgame/visual-novel scripts (including H-scenes) using the hymt2-heretic-colab Google Colab L4 + Google Drive workflow via drive_bridge.py (replaces vast.ai).
---

# Hy-MT2 Colab Translate Skill

Follow `deploy/colab-hymt2/AGENT_SOP_COLAB.md` strictly when handling English -> Taiwan Traditional Chinese adult Galgame script translation on branch `hymt2-heretic-colab`.

## Mandatory Rules
1. **Never** run `vastai`, ask for `VAST_API_KEY`, or create Windows scheduled tasks (`HanhuaGuard`).
2. **Never** copy files directly into `G:\` or `H:\` by hand; always use `deploy/colab-hymt2/drive_bridge.py` (or `<instructions_dir>/drive_bridge.py` when working from a standalone game workspace).
3. **Never** upload the 18.24 GB GGUF from local disk; instruct the user to run `deploy_models_colab.ipynb` on Free CPU Colab if `doctor` reports a missing model.
4. Keep `ui`/`system` lines marked `"kind": "ui"` or `"system"` so adult style and story context are suppressed, and preserve all placeholders (`{{TAG_n}}`, `⟦n⟧`, `\C[n]`).
5. Keep original game files untouched and never distribute outputs publicly.
6. After `drive_bridge.py pull` succeeds (SHA-256 + ID parity verified) and local files are confirmed saved, delete that finished job's residuals in `hymt2/input/`, `hymt2/work/`, and `hymt2/output/` so the next job never scans stale inputs — **NEVER delete `models/`, `bin/`, or `scripts/`**.

## Standard Commands
```powershell
# (If working outside the repo, invoke <instructions_dir>/drive_bridge.py instead of deploy/colab-hymt2/drive_bridge.py)
# 0. Check Drive mount and model readiness
python deploy/colab-hymt2/drive_bridge.py doctor

# 1. Validate strict JSONL and optional glossary locally
python deploy/colab-hymt2/drive_bridge.py validate --input <job.jsonl> [--glossary <glossary.json>]

# 2. Push atomically + manifest + runner scripts to Google Drive
python deploy/colab-hymt2/drive_bridge.py push --input <job.jsonl> [--glossary <glossary.json>] --name <job>
# -> STOP here and instruct user to run latest hymt2_colab.ipynb on Google Colab L4 (Ctrl+F9).

# 3. After user confirms Colab finished (either rc=0 [DONE] or rc=2 [PARTIAL]), check QA/DONE:
python deploy/colab-hymt2/drive_bridge.py status --name <job>

# 3a. Rebuild from checkpoint + <job>.overrides.jsonl with ZERO Colab GPU usage and no model calls:
python <instructions_dir>/runner.py --drive-root "<drive_root>" --assemble-only --job <job> --manifest-wait 0
#     To have the model retry: run the latest notebook with REPAIR_ONLY=True & TARGET_JOB="<job>".

# 3b. [DONE] can still contain untranslated English rows: always report <job>.review.jsonl to the user.
#     Pull verified files, confirm they are saved locally, and only then clean up hymt2/{input,work,output}/.
python deploy/colab-hymt2/drive_bridge.py pull --name <job> --dest <out_dir>
```
