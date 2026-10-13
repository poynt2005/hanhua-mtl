# 03 — CLI and configuration reference

## Commands

| Command | Purpose |
|---|---|
| `auto` | one-shot: archive in → `<stem>_MTL_zhTW<ext>` out (extract→translate→verify→render→repack) |
| `book` | same as `auto` but simpler defaults; `-o` optional |
| `unpack` | extract an archive's image entries to a folder |
| `repack` | rebuild an archive from a folder of rendered pages |
| `extract` | detect + OCR a folder into `project.json` |
| `translate` | translate a `project.json` into `project.zh.json` |
| `render` | erase + typeset a project into a folder |
| `run` | extract + translate + render a folder (no repack) |
| `terms` | extract candidate terms and propose a glossary |
| `normalize` | enforce the glossary in existing translations |
| `verify` | check glossary consistency (non-zero exit on violation) |
| `ping` | test the translation endpoint |
| `selftest` | offline end-to-end test with the mock provider |
| `vast` | vast.ai lifecycle: `status` / `search` / `up` / `down` |

### Common flags

| Flag | Meaning |
|---|---|
| `--config PATH` | config file (per subcommand) |
| `--profile {openai,sakura}` | translation prompt profile |
| `--model NAME` | override the text model name |
| `--api-base`, `--api-key` | override the endpoint / key |
| `--glossary PATH` | glossary file to enforce |
| `--timeout SECONDS` | HTTP timeout |
| `--mock` | use the offline mock provider |
| `--font PATH` | CJK font path |

### `auto`

```
hanhua.py auto -i INPUT [-o OUTPUT] [--ocr BACKEND] [--inpaint METHOD]
             [--glossary FILE] [--auto-terms] [--strict] [--keep-work]
```
- `--ocr`: `vision` `http` `manga_ocr` `opencv` `sidecar` `mock` (default `http`)
- `--inpaint`: `textmask` `fill` `blur` `opencv` `lama`
- `--auto-terms`: build a best-effort name glossary
- `--strict`: abort if the glossary is inconsistent (default: warn and continue)
- `--keep-work`: keep the temp work dir even on success (on failure it is always kept)

Default output name: `<input stem>_MTL_zhTW<ext>`.

### `vast`

```
hanhua.py vast status
hanhua.py vast search [--min-down N] [--min-up N] [--max-price F] [--gpu NAME]
hanhua.py vast up --yes [--attempts N] [--image TAG] [--baked] [--write-config FILE]
hanhua.py vast down
```
- `status`: account credit/spend + current instances.
- `search`: list offers (cheapest within a reliability band).
- `up`: rent + deploy + return endpoints. Requires `--yes`. With `--baked` and an
  `--image` that already contains the stack, no install runs on the box.
- `down`: destroy all instances and verify.

## Configuration (`config.example.json`)

| Field | Meaning |
|---|---|
| `profile` | `openai` (any OpenAI-compatible LLM) or `sakura` (self-hosted) |
| `api_base` / `api_key` / `text_model` | translation endpoint |
| `vision_model` | model for the `vision` OCR backend |
| `source_lang` / `target_lang` | languages (default `ja` → `zh-TW`) |
| `system_prompt` | override the translation system prompt |
| `glossary_file` | glossary to apply automatically |
| `ocr_url` / `ocr_prepare_url` / `ocr_api_key` | self-hosted OCR+inpaint service |
| `font` | CJK font path (auto-detected if empty) |
| `inpaint` / `padding` / `dilate` | erase method and mask tuning |
| `stroke` | text outline width |
| `translate_chunk` / `translate_max_lines` | translation batching |
| `keep_sfx` | keep `type: sfx` blocks as-is |
| `convert_tw` | convert simplified output to Traditional (needs OpenCC) |
| `timeout` | HTTP timeout |
| `pattern` | image glob when reading a folder |
| `output_format` / `keep_ext` | output format; `keep_ext` preserves the source extension |

### Environment variables

| Variable | Meaning |
|---|---|
| `MTH_API_KEY` | API key |
| `MTH_API_BASE` | API base URL |
| `MTH_TEXT_MODEL`, `MTH_VISION_MODEL` | model names |
| `MTH_SOURCE_LANG`, `MTH_TARGET_LANG` | languages |
| `MTH_FONT` | font path |
| `VAST_API_KEY` | vast.ai API key (never written to a file) |

## Examples

```bash
# offline dry run
python hanhua.py auto -i book.cbz --ocr mock

# self-hosted GPU stack
python hanhua.py vast up --yes --image <registry>/hanhua-gpu:1 --baked --write-config config.json
python hanhua.py auto -i book.cbz --ocr http --profile sakura --inpaint lama

# cloud LLM
MTH_API_KEY=... python hanhua.py auto -i book.cbz --ocr http --profile openai

# teardown
python hanhua.py vast down
```
