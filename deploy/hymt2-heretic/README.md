# hymt2-heretic -- uncensored EN->ZH translation endpoint for adult galgames

A lean GPU image: the official llama.cpp CUDA server plus an uncensored
translation model, for **English -> Chinese** galgame text including H-scenes
(explicit vocabulary, onomatopoeia, moaning). Boots make **no network calls**.

| | |
|---|---|
| Built by | `.github/workflows/build-hymt2-image.yml` (branch `hymt2-heretic`) |
| Pushed to | `docker.io/<DOCKERUSR>/hanhua-gpu:hymt2-30b-a3b-heretic-q4km` |
| Model | `hy-mt2-heretic` = Hy-MT2-30B-A3B-uncensored-heretic, imatrix Q4_K_M, 18.24 GB, sha256-pinned |
| Server | llama.cpp `server-cuda-b12019` (CUDA 12.8), OpenAI-compatible `:11434/v1` |
| Slots / context | 4 parallel requests x 8192 tokens (`LLAMA_ARG_N_PARALLEL`, `LLAMA_ARG_CTX_SIZE`) |

## Why this model

The galgame line on `sakura-ollama` (Sakura-GalTransl) is **Japanese**->Chinese
only; it cannot serve English source text. For EN->ZH:

- **Base: tencent/Hy-MT2-30B-A3B** (2026-05) -- a dedicated translation model
  (not a chat model), and English<->Chinese is its strongest direction.
  MoE with 3B active parameters, so it generates far faster than a 14B dense
  model; Apache-2.0.
- **Decensored: OS-Software's Heretic v1.4 (ARA)** -- the original refuses this
  material (100/100 on the author's set); this one 0/100, with KL divergence
  0.0276 from the original, i.e. translation ability essentially untouched.
- **Quant: i1-Q4_K_M** -- the imatrix build marked "recommended"; the largest
  that leaves KV-cache headroom on a 24 GB card (Q6_K is 24.8 GB and does not fit).

No open model is trained specifically on adult galgame text in either language
pair (searched 2026-11-18). Decensoring removes refusals; it does not teach a
house style for explicit terms or SFX. Control that with the terminology and
style prompts below.

## Host requirements

- **24 GB VRAM** (RTX 3090 / 4090 / A5000 / L4). 16 GB cards do not fit this quant.
- **NVIDIA driver new enough for CUDA 12.8** (on vast.ai filter `cuda_max_good >= 12.8`).
  An older driver fails at container start, not at build.

## Run and check

```bash
docker run --gpus all -p 11434:11434 <DOCKERUSR>/hanhua-gpu:hymt2-30b-a3b-heretic-q4km
curl http://localhost:11434/health          # {"status":"ok"} once loaded
curl http://localhost:11434/v1/models       # lists "hy-mt2-heretic"
```

Gate before any batch: one request below must return a single line of Chinese
with no explanation, no `<think>` block and no refusal.

## How to prompt it (from the model card)

- **No system prompt.** Put everything in one user message.
- No thinking: the chat template defaults to `reasoning_effort = no_think`;
  do not pass `high`/`low`.
- Sampling: `temperature 0.7, top_p 1.0, top_k off, repetition_penalty 1.0,
  max_tokens 4096` (already the server defaults; a client may override).
- Chinese instructions must name the language in full Chinese: `繁体中文`.

Default:

```
将以下文本翻译为繁体中文，注意只需要输出翻译后的结果，不要额外解释：

{source_text}
```

Terminology (keeps explicit terms and SFX consistent across a script -- feed
the glossary for every chunk):

```
参考下面的翻译：
{en_term} 翻译成 {zh_term}
{en_term} 翻译成 {zh_term}

将以下文本翻译为繁体中文，注意只需要输出翻译后的结果，不要额外解释：

{source_text}
```

Style:

```
请将以下文本翻译为繁体中文。
注意翻译的风格要严格符合【{target_style}】
{source_text}
```

Line-aligned batches (several script lines in one request): use the card's
delimiter template -- "你必须在译文中保留等量的分隔符，绝对不可遗漏、转义或翻译该符号，并注意分隔符的位置" --
and split on the delimiter; fall back to one line per request on a count
mismatch, as the Sakura client does.

## Client changes needed

The galgame client's `--profile galtransl` speaks the **Sakura** prompt format
(system prompt + "历史翻译/术语表" user block). That format is wrong for this
model. Add a `hymt2` profile that sends the templates above with no system
message, and point it at model `hy-mt2-heretic`.

## Known risks (unmeasured until the first run)

1. The decensor was measured on **Japanese** prompts; English refusal rate is
   expected to follow but has not been measured.
2. `hy_v3` support in llama.cpp is recent (b9993); loading this exact GGUF on
   b12019 is expected to work but is verified only by the first boot.
3. Decensored != stylistically explicit: if the output euphemises, fix it with
   the terminology/style templates before considering another model.

## Rebuild / change the model

`workflow_dispatch` the workflow, or change `HF_REPO` / `HF_REV` / `GGUF_FILE` /
`GGUF_SHA256` in the Dockerfile **and pick a new tag** -- never `latest`.
