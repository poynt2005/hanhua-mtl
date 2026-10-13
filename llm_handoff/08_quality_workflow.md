# 08 — Quality workflow (terms, names, consistency)

## Why

The same character name or term must read the same everywhere. Machine
translation drifts (different renderings of the same source word), so the tool
provides an explicit glossary workflow.

## The loop

```
terms  ->  human/agent review  ->  translate --glossary  ->  normalize  ->  verify
```

| Step | Command | What it does |
|---|---|---|
| extract candidates | `terms -p project.json -o glossary.json --min-count 2` | finds recurring tokens (katakana/kanji/latin runs) and proposes translations |
| review | (edit the file) | a human or agent sets one canonical target per source term |
| translate | `translate --glossary glossary.json` | injects the glossary into the translation request |
| normalize | `normalize --glossary glossary.json` | rewrites translations that deviate from the glossary |
| verify | `verify --glossary glossary.json` | lists violations; non-zero exit if any |

`auto` runs translate + normalize + verify when a glossary is supplied, and by
default **warns and continues** on violations (so a run is not lost); `--strict`
aborts instead.

## Glossary file format

```json
{
  "version": 1,
  "target_lang": "zh-TW",
  "terms": [
    { "src": "<source>", "zh": "<canonical>", "type": "name", "count": 9, "status": "approved" }
  ]
}
```

## Practical notes learned

- **Propose, then review.** Automatic proposals are a starting point; the agent's
  job is to guarantee "one source term → one target".
- **Keep only proper nouns.** Common words should not be locked (it over-
  constrains natural phrasing).
- **Do not use the translation model for JSON extraction.** It returns schema
  keys instead of data. Generate the glossary by translating the candidate terms
  line-by-line (which the model is good at) and then review.
- **Word-order variants are normal.** If a target only differs by natural word
  order in context, adjust the glossary target rather than forcing an awkward
  phrasing.
- **Watch for dropped names.** A short line may be translated without the
  character name; `verify` catches this.

## Translation prompt

For the self-hosted visual-novel-tuned profile the tool uses the model's tuned
system prompt and a user message that includes:

1. recent context (previous translated lines),
2. the glossary,
3. the current batch of source lines.

Sampling is set to the model's recommended values, and batches are kept small
(about ten lines) — large batches without context are what produced literal,
"machine-translation-looking" output.

## Simplified → Traditional

The self-hosted model outputs simplified Chinese. With `convert_tw` enabled the
tool converts to Traditional (Taiwan) using OpenCC (`s2twp`). If OpenCC is not
installed, the conversion is skipped.

## Honest limitation

The consistency tooling is only as good as the glossary. For a short work with a
small cast, a handful of names is enough; for a long series, build the glossary
from the first chapters and reuse it across volumes.
