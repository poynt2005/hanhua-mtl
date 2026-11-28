---
name: sakura-colab-translate
description: Use when translating Japanese Galgame/visual-novel scripts using the sakura-ollama-colab Google Colab L4 + Google Drive workflow via drive_bridge.py (replaces vast.ai).
---

# Sakura Colab Translate Skill

Follow `deploy/colab-sakura/AGENT_SOP_COLAB.md` strictly when handling Japanese -> Traditional Chinese Galgame script translation on branch `sakura-ollama-colab`.

## Mandatory Rules
1. **Never** run `vastai`, ask for `VAST_API_KEY`, or create Windows scheduled tasks (`HanhuaGuard`).
2. **Never** copy files directly into `G:\` or `H:\` by hand; always use `deploy/colab-sakura/drive_bridge.py` (or `<instructions_dir>/drive_bridge.py` when working from a standalone game workspace).
3. **Never** upload GGUF weights from local disk; instruct the user to run `deploy_models_colab.ipynb` on Free CPU Colab if `doctor` reports a missing model.
4. Preserve all placeholders (`{{TAG_n}}`, `⟦n⟧`, `\C[n]`) and keep `ui`/`system` lines marked `"kind": "ui"` or `"system"` so history is disabled.
5. Keep original game files untouched and never distribute outputs publicly.
6. After `drive_bridge.py pull` succeeds (SHA-256 + ID parity verified) and local files are confirmed saved, delete that finished job's residuals in `sakura/input/`, `sakura/work/`, and `sakura/output/` so the next job never scans stale inputs — **NEVER delete `models/`, `bin/`, or `scripts/`**.

## Standard Commands
```powershell
# (If working outside the repo, invoke <instructions_dir>/drive_bridge.py instead of deploy/colab-sakura/drive_bridge.py)
# 0. Check Drive mount and model readiness
python deploy/colab-sakura/drive_bridge.py doctor

# 1. Validate strict JSONL and glossary locally
python deploy/colab-sakura/drive_bridge.py validate --input <job.jsonl> [--glossary <glossary.json>]

# 2. Push atomically + manifest + runner scripts to Google Drive
python deploy/colab-sakura/drive_bridge.py push --input <job.jsonl> [--glossary <glossary.json>] --name <job>
# -> STOP here and instruct user to run latest sakura_colab.ipynb on Google Colab L4 (Ctrl+F9).

# 3. After user confirms Colab finished (either rc=0 [DONE] or rc=2 [PARTIAL]), check QA/DONE:
python deploy/colab-sakura/drive_bridge.py status --name <job>

# 3a. To rebuild from the checkpoint + <job>.overrides.jsonl with ZERO Colab GPU usage and no model calls
#     (writes .DONE.json if rows still in Japanese are <= 0.1% of all rows):
python <instructions_dir>/runner.py --drive-root "<drive_root>" --assemble-only --job <job> --manifest-wait 0
#     To have the model retry, add glossary/partial overrides and instruct user to run Colab with REPAIR_ONLY=True & TARGET_JOB="<job>".
#     [DONE] can still contain Japanese or rule-generated rows: always report <job>.review.jsonl to the user.

# 3b. Once DONE verified (or to retest post-DONE: use overrides + --repair-only, or delete target rows in checkpoint.jsonl + FORCE_RERUN=True),
#     pull verified files, verify saved locally, and only then clean up sakura/{input,work,output}/ for that job.
python deploy/colab-sakura/drive_bridge.py pull --name <job> --dest <out_dir>
```
