---
name: sakura-colab-translate
description: Use when translating Japanese Galgame/visual-novel scripts using the sakura-ollama-colab Google Colab L4 + Google Drive workflow via drive_bridge.py (replaces vast.ai).
---

# Sakura Colab Translate Skill

Follow `deploy/colab-sakura/AGENT_SOP_COLAB.md` strictly when handling Japanese -> Traditional Chinese Galgame script translation on branch `sakura-ollama-colab`.

## Mandatory Rules
1. **Never** run `vastai`, ask for `VAST_API_KEY`, or create Windows scheduled tasks (`HanhuaGuard`).
2. **Never** copy files directly into `G:\` or `H:\` by hand; always use `deploy/colab-sakura/drive_bridge.py`.
3. **Never** upload GGUF weights from local disk; instruct the user to run `deploy_models_colab.ipynb` on Free CPU Colab if `doctor` reports a missing model.
4. Preserve all placeholders (`{{TAG_n}}`, `⟦n⟧`, `\C[n]`) and keep `ui`/`system` lines marked `"kind": "ui"` or `"system"` so history is disabled.
5. Keep original game files untouched and never distribute outputs publicly.

## Standard Commands
```powershell
# 0. Check Drive mount and model readiness
python deploy/colab-sakura/drive_bridge.py doctor

# 1. Validate strict JSONL and glossary locally
python deploy/colab-sakura/drive_bridge.py validate --input <job.jsonl> [--glossary <glossary.json>]

# 2. Push atomically + manifest + runner scripts to Google Drive
python deploy/colab-sakura/drive_bridge.py push --input <job.jsonl> [--glossary <glossary.json>] --name <job>
# -> STOP here and instruct user to run sakura_colab.ipynb on Google Colab L4 (Ctrl+F9).

# 3. After user confirms Colab finished, check QA/DONE and pull verified files
python deploy/colab-sakura/drive_bridge.py status --name <job>
python deploy/colab-sakura/drive_bridge.py pull --name <job> --dest <out_dir>
```
