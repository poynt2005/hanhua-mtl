---
name: hymt2-colab-translate
description: Use when translating English adult Galgame/visual-novel scripts (including H-scenes) using the hymt2-heretic-colab Google Colab L4 + Google Drive workflow via drive_bridge.py (replaces vast.ai).
---

# Hy-MT2 Colab Translate Skill

Follow `deploy/colab-hymt2/AGENT_SOP_COLAB.md` strictly when handling English -> Taiwan Traditional Chinese adult Galgame script translation on branch `hymt2-heretic-colab`.

## Mandatory Rules
1. **Never** run `vastai`, ask for `VAST_API_KEY`, or create Windows scheduled tasks (`HanhuaGuard`).
2. **Never** copy files directly into `G:\` or `H:\` by hand; always use `deploy/colab-hymt2/drive_bridge.py`.
3. **Never** upload the 18.24 GB GGUF from local disk; instruct the user to run `deploy_models_colab.ipynb` on Free CPU Colab if `doctor` reports a missing model.
4. Keep `ui`/`system` lines marked `"kind": "ui"` or `"system"` so adult style and story context are suppressed, and preserve all placeholders (`{{TAG_n}}`, `⟦n⟧`, `\C[n]`).
5. Keep original game files untouched and never distribute outputs publicly.

## Standard Commands
```powershell
# 0. Check Drive mount and model readiness
python deploy/colab-hymt2/drive_bridge.py doctor

# 1. Validate strict JSONL and optional glossary locally
python deploy/colab-hymt2/drive_bridge.py validate --input <job.jsonl> [--glossary <glossary.json>]

# 2. Push atomically + manifest + runner scripts to Google Drive
python deploy/colab-hymt2/drive_bridge.py push --input <job.jsonl> [--glossary <glossary.json>] --name <job>
# -> STOP here and instruct user to run hymt2_colab.ipynb on Google Colab L4 (Ctrl+F9).

# 3. After user confirms Colab finished, check QA/DONE and pull verified files
python deploy/colab-hymt2/drive_bridge.py status --name <job>
python deploy/colab-hymt2/drive_bridge.py pull --name <job> --dest <out_dir>
```
