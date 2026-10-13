# 10 — Current state and next steps

## Exactly where things stand

| Item | State |
|---|---|
| Core pipeline (`unpack→extract→translate→render→repack`) | **works**, verified end to end on a sample archive |
| `auto` one-shot command | works; default output `<stem>_MTL_zhTW<ext>` |
| OCR/inpaint service (`deploy/ocr_server.py`) | works; text-based mask, LaMa erase, crops back to original size |
| Local CPU path | works; inpainting limited to lightweight methods (no LaMa without torch) |
| GPU image (`deploy/Dockerfile`) | **written and syntax-checked; NOT built or verified** |
| Container start script | written and syntax-checked |
| CI workflow (build + push) | written (uses the repository's Docker Hub secrets) |
| vast.ai controller (`mth/vast.py`) | works; happy path is HTTP-only (no SSH); leftovers are cleaned; failures destroy the machine |
| Cost-safety docs | playbook + skill + hard rules in place |
| Term consistency | implemented (extract / review / translate / normalize / verify) |
| Sample deliverable | produced; **gitignored** (never committed) |

## Known issues in the existing sample deliverable

1. **Dimensions were padded** (+5/+7 px) because the run happened before the
   "crop the inpainting result back to the original size" fix. A re-run with the
   current code produces exact dimensions.
2. **Erase quality was not formally reviewed by the user.** The erase switched
   from colour filling to neural inpainting, but the user has not confirmed the
   result.

## Prioritised next steps

### P0 — build and validate the image
- Build `deploy/Dockerfile` (locally with Docker, or via the CI workflow).
- The build-time smoke test must pass. Fix any dependency issue it reports.
- Push to a registry the GPU host can pull from.

### P0 — re-run the sample with the baked image
- `vast up --yes --image <tag> --baked --write-config config.json`
- `auto -i <archive> --ocr http --profile sakura --inpaint lama`
- Verify exact dimensions and page order; **have the user review erase and
  translation quality** before destroying anything else.

### P1 — quality follow-ups (only if the user wants them)
- Use a general instruction-following LLM for `terms`/`normalize` (the translation
  model is unreliable at JSON).
- Improve the auto-glossary heuristic to catch kanji names.
- Better vertical-text fitting; wrap Latin text at word boundaries.

### P1 — image hardening
- Pin the base image by digest.
- Explicitly download/verify the segmentation detector weights, or assert it
  loads at build time.

### P2 — nice to have
- Retry per page in the HTTP OCR client.
- Optional PSD export for manual touch-up.

## Rules to keep following

- Never commit archives, pipeline artifacts, `config.json`, keys, or any input
  content (see `.gitignore`).
- Keep the money-first reporting habit and the $1 cap when renting.
- Keep the machine until the output is confirmed; reuse it for fixes instead of
  re-renting.

## If you are picking this up

1. Read `INDEX.md`, then `01`→`11`.
2. Run `selftest` and `auto --ocr mock` to confirm the pipeline.
3. Build the image and run one real pass.
4. Ask the user to review the output before spending more on machines.
