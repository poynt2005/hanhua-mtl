# hymt2-heretic -- uncensored EN->ZH translation endpoint for adult galgames

A lean GPU image: the official llama.cpp CUDA server plus an uncensored
translation model, for **English -> Chinese** galgame text including H-scenes
(explicit vocabulary, onomatopoeia, moaning). The model translates into
**Simplified** (its strongest Chinese); the client converts to Taiwan
Traditional with OpenCC `s2twp`. Boots make **no network calls**. It serves the
separate galgame pipeline, not hanhua's manga pipeline.

| | |
|---|---|
| Image | `docker.io/<DOCKERUSR>/hanhua-gpu:hymt2-30b-a3b-heretic-q4km` |
| First published | 2026-11-18, digest `sha256:06bc4600bdded260c017758bfb4ca8f600ecc5f849f6b385082c9d33b23f975a` |
| Built by | `.github/workflows/build-hymt2-image.yml` (branch `hymt2-heretic` only) |
| Model | `hy-mt2-heretic` = Hy-MT2-30B-A3B-uncensored-heretic, imatrix Q4_K_M, 18.24 GB, commit- and sha256-pinned |
| Server | llama.cpp `server-cuda-b12019` (CUDA 12.8), OpenAI-compatible on `:11434/v1` |
| Slots / context | 4 parallel requests, **8192 tokens each** (32768 shared) |
| Reference client | `hymt2_prompt.py` (stdlib only): the prompt, a streaming client, the gate |
| Handoff | `llm_handoff/19_galgame_image_branches.md`: decision record, build record, open items |
| State (2026-11-18) | built and pushed; **never booted on a GPU yet**; `hanhua.py vast up` cannot launch it as-is (section 3.1) |

---

## 1. Why this model

- The `sakura-ollama` image (Sakura-GalTransl) is **Japanese**->Chinese only.
- **tencent/Hy-MT2-30B-A3B** (2026-05) is a dedicated translation model;
  English<->Chinese is its strongest direction; MoE with 3B active parameters,
  so it is fast; Apache-2.0.
- **Simplified out, Traditional by OpenCC** (section 6.1): Simplified is where
  the model's Chinese data -- and its slang/SFX vocabulary -- is; `s2twp` then
  gives Taiwan characters *and* Taiwan vocabulary deterministically, which a
  "translate into Traditional" instruction does not.
- **Heretic v1.4 (ARA) by OS-Software** removes refusals: 100/100 -> 0/100 on
  the author's (Japanese) set, KL divergence 0.0276 from the original.
- **i1-Q4_K_M**: the imatrix quant marked "recommended"; the largest that fits a
  24 GB card with KV headroom (Q6_K is 24.8 GB).
- No open model is trained specifically on adult galgame text (searched
  2026-11-18). The general alternative is an abliterated ~27B chat model:
  better at long context, several times slower, not trained for faithful,
  line-aligned translation. Switch only if this model measurably fails.

## 2. What differs from the sakura-ollama image

| | sakura-ollama | hymt2-heretic | Consequence for a client |
|---|---|---|---|
| Direction | JA -> Simplified | **EN -> Simplified** | keep an OpenCC step, but use **`s2twp`** (Taiwan vocabulary) |
| Prompt | Sakura system prompt + "历史翻译/术语表" block | **no system prompt**; Hy-MT2 templates (section 6) | must change |
| Sampling | temp 0.3, top_p 0.8 | temp 0.7, top_p 1.0, top_k off | send explicitly; client values override the server |
| Context per request | 16384 | **8192** | keep history short (the reference sends 3 lines) |
| Model name | `sakura-galtransl-14b` | `hy-mt2-heretic` | config |
| Server | ollama | llama.cpp | **no `/api/*`** (no `/api/tags`, no `keep_alive: 0` unload); use `/v1/*` and `/health` |
| Driver | any recent | **CUDA 12.8 capable** | filter hosts (section 3) |
| Healthcheck | none | Docker `HEALTHCHECK` on `/health` | `docker ps` shows `healthy` once loaded |
| Base image | `ollama/ollama:latest` (floating) | pinned `server-cuda-b12019` | reproducible |
| Same | port 11434, OpenAI `/v1`, 4 parallel slots, model always resident, offline boot | | |

## 3. Host requirements

- **24 GB VRAM** (RTX 3090 / 4090 / A5000 / L4). 16 GB cards do not fit.
  Budget: weights ~17.0 GiB + KV 3.0 GiB (96 KiB/token x 32768) + buffers ~= 21 GiB.
- **Driver that supports CUDA 12.8** (R570+). On vast.ai filter
  `cuda_max_good >= 12.8`. An older driver fails at container start.
- Disk: image ~21 GB.
- vast.ai: read `deploy/VASTAI_PLAYBOOK.md` first (budget before renting, one
  machine, $1 cap, verify teardown). hanhua's `vast up --baked` and
  `tools/gate.py` assume the master image (ports 8000/8080, ollama API) and do
  **not** apply; map port 11434 and use `hymt2_prompt.py gate`.

### 3.1 On vast.ai: the image's CMD does not run

hanhua rents with `--ssh` launch mode. In that mode vast runs **its own**
entrypoint (sshd), so this image's `ENTRYPOINT` / `CMD` never start the server:
the onstart command must. Master's `hanhua.py vast up` today maps ports
8000/8080, filters `cuda_max_good >= 12.1`, runs the master image's
`container_start.sh` and waits for 8000 + 8080 -- all wrong for this image. Fix
that first (free, `--dry-run` testable; see
`llm_handoff/19_galgame_image_branches.md` section 5): port 11434,
`cuda_max_good >= 12.8`, this onstart, readiness on `:11434/health`.

```bash
nohup /app/llama-server -m /models/model.gguf --alias hy-mt2-heretic \
  --host 0.0.0.0 --port 11434 -ngl 999 -c 32768 -np 4 --jinja \
  --temp 0.7 --top-p 1.0 --top-k 0 --repeat-penalty 1.0 \
  >/root/llama.log 2>&1 </dev/null & echo started
```

Flags are explicit so the onstart does not depend on vast passing the image's
`ENV` through. If the binary cannot find its shared libraries, prefix
`LD_LIBRARY_PATH=/app`. The log is `/root/llama.log`.

## 4. Run and configure

```bash
docker run -d --gpus all -p 11434:11434 --name hymt2 \
    <DOCKERUSR>/hanhua-gpu:hymt2-30b-a3b-heretic-q4km
```

Override any server setting with `-e` (llama-server reads `LLAMA_ARG_*`):

| Variable | Default | Meaning |
|---|---|---|
| `LLAMA_ARG_CTX_SIZE` | `32768` | total context, shared by the slots |
| `LLAMA_ARG_N_PARALLEL` | `4` | concurrent requests; per-request context = CTX_SIZE / N_PARALLEL |
| `LLAMA_ARG_N_GPU_LAYERS` | `999` | all layers on GPU |
| `LLAMA_ARG_PORT` | `11434` | listen port (the healthcheck assumes 11434) |
| `LLAMA_ARG_ALIAS` | `hy-mt2-heretic` | name reported by `/v1/models` |
| `LLAMA_ARG_UI` | `0` | web UI off |

Sampling defaults are set on the command line (`--temp 0.7 --top-p 1.0
--top-k 0 --repeat-penalty 1.0`); request parameters override them.

## 5. Boot check and gate (before any batch)

```bash
curl http://<host>:11434/health        # {"status":"ok"}; 503 while loading (~1-3 min)
curl http://<host>:11434/v1/models     # lists "hy-mt2-heretic"
python hymt2_prompt.py --api-base http://<host>:11434/v1 gate
```

The gate translates a plain line, an explicit line and an SFX line. Each must
return one non-empty Chinese line with no refusal, no explanation, no thinking
block; exit code 1 otherwise. **Do not run a batch on a failed gate.**

## 6. Prompting (required -- the model has no default system prompt)

Rules from the model card:

- No system message. One user message.
- Chinese instructions (Simplified, as in the card); name the target in full:
  `简体中文`.
- No thinking: the chat template defaults to `reasoning_effort = no_think`; do
  not pass `high`/`low`.

The prompt this image ships with (`hymt2_prompt.build_messages`) uses the card's
**personalization** template -- the only official one that carries several
requirements at once. Example for `Fuck me harder, Onii-chan!` with a glossary
(kept in Traditional: `"harder": "用力幹"`) and a speaker (`優衣`):

```
【待翻译文本】
Fuck me harder, Onii-chan!
【翻译任务】
1、成人向Galgame台词风格：口语化、直白露骨，性器官与性行为使用直接的说法，不委婉、不省略、不说教；拟声词与喘息译成对应的中文拟声词，保留原文的拉长音、重复次数与标点（……、～、！）
2、参考下面的翻译：onii-chan 翻译成 哥哥；harder 翻译成 用力干
3、说话人是「优衣」，人名保持一致
4、上文（仅供参考，不要翻译）：<previous 3 lines>
5、只输出译文，不要额外解释，不要输出原文
6、将【待翻译文本】翻译为简体中文。
```

- **Line 1 (style)** is what stops euphemising: decensoring removes refusals,
  not prudish word choice. Edit `STYLE_ADULT` to set the house style.
- **Glossary**: a JSON `{"english term": "中文"}`. Only terms that occur in the
  line (case-insensitive, whole word) are sent, so the prompt stays short. Put
  names, honorifics, explicit terms and recurring SFX here -- it is the
  mechanism for one source term -> one translation across a script.
- **Context**: at most the 3 previous lines, marked "do not translate".
- **One script line per request.** 4 slots x 3B active parameters make this
  fast, and line alignment can then never break. Do not batch lines with
  newlines.
- **Fallback**: if the output fails the mechanical checks (empty, thinking tag,
  "译文：" prefix, extra lines, untranslated), `translate()` retries once with
  the card's plain template:
  `将以下文本翻译为简体中文，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}`
- **Stream every request** and retry dropped connections: a rented host's port
  forward drops a silent socket after ~15 s.

### 6.1 Script: the model works in Simplified only

Goal: the model's side runs at full capability; Traditional is a post-process.

- **Target `简体中文`.** The model's Chinese training data, and most of the
  explicit/SFX vocabulary it knows, is Simplified. Asking for `繁体中文` makes the
  model do two jobs at once (translate + convert script) and still yields
  mainland vocabulary in Traditional glyphs, with occasional Simplified leaks.
- **One script in the prompt.** Glossary targets and speaker names are converted
  to Simplified (`tw2sp`) before they are sent, so a glossary can be maintained
  in Traditional without mixing scripts in what the model sees.
- **Convert after**, client side: OpenCC **`s2twp`** = Taiwan characters + Taiwan
  phrases (软件->軟體, 视频->影片), context-aware for one-to-many characters
  (干净->乾淨, 干我->幹我, 头发->頭髮). `translate()` does this by default;
  `traditional=False` returns the raw Simplified.
- Requires `pip install opencc-python-reimplemented` on the client. Without it
  `to_traditional()` raises; it never silently returns Simplified.
- Spot-check one-to-many characters (干/幹/乾, 发/發/髮, 后/後/后, 面/麵) in the
  first real batch; fix a recurring miss by putting the phrase in the glossary.
- What else keeps the model at full strength: the card's sampling (sent per
  request), `no_think`, its trained templates. The one capability trade-off in
  this image is the quant: **Q4_K_M** is the largest that fits 24 GB; Q6_K
  (24.8 GB) or Q8_0 (32.1 GB) need a 32-48 GB card and a new image tag.

## 7. Client integration checklist (the galgame client's `hymt2` profile)

- [ ] model `hy-mt2-heretic`, base `http://<host>:<mapped 11434>/v1`
- [ ] prompt from `hymt2_prompt.build_messages` (or call `translate()`); no system message
- [ ] sampling `temperature 0.7, top_p 1.0, top_k 0, repeat_penalty 1.0` sent per request
- [ ] `stream: true` + retries
- [ ] one line per request; up to 4 in flight
- [ ] target `简体中文` in the prompt; glossary/speaker converted to Simplified before sending
- [ ] OpenCC **`s2twp`** on the output (not `s2t`/`s2tw`: those skip Taiwan vocabulary)
- [ ] no ollama `/api/*` calls (unload, tags); readiness via `/health`
- [ ] keep any history/context within 8192 tokens per request
- [ ] glossary file for names, honorifics, explicit terms, SFX

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| container exits at start, CUDA driver error | host driver older than CUDA 12.8 | pick a host with `cuda_max_good >= 12.8` |
| `/health` 503 for minutes | still loading 18 GB from disk | wait; slow disks take longer than the 180 s start period |
| CUDA out of memory at load | card < 24 GB, or another process on the GPU | 24 GB card; or `-e LLAMA_ARG_CTX_SIZE=16384` |
| `404` on `/api/generate` / `/api/tags` | client speaks ollama's native API | use `/v1/*`; drop unload calls (the model stays resident) |
| reply is an explanation / has "译文：" | rich prompt misread | `translate()` already falls back; if frequent, shorten `STYLE_ADULT` |
| euphemised wording | model's prior word choice | add the terms to the glossary; tighten `STYLE_ADULT` |
| refusal text | decensor measured on Japanese, not English | report it with the input line; try the plain template |
| `context size exceeded` | request > 8192 tokens | less context, or fewer slots (`LLAMA_ARG_N_PARALLEL=2` -> 16384 each) |
| connection dropped mid-reply | port-forward idle timeout | stream (default in `hymt2_prompt.py`) |
| final output still Simplified | OpenCC step skipped / not installed | `pip install opencc-python-reimplemented`; use `translate()` or `to_traditional()` |
| mainland wording in the final text (软件 -> 軟件) | converted with `s2t`/`s2tw` | use `s2twp` |
| wrong one-to-many character (乾我, 頭發) | OpenCC phrase table miss | add the phrase to the glossary; report the line |
| model output partly Traditional | Traditional text inside the prompt | convert glossary/speaker with `to_simplified()` (the reference does) |

## 9. Known risks (unmeasured until the first GPU run)

1. Loading this GGUF on b12019 is expected (architecture `hy_v3` is in that
   build) but proven only by the first boot.
2. The refusal rate was measured on Japanese prompts; English is expected to
   follow.
3. The personalization prompt with an explicit style line is this image's
   design, not a card example; the gate and a 50-line spot check on real H-scene
   text must confirm it before a full script.
4. Starting `/app/llama-server` from a vast onstart (section 3.1) instead of
   the image entrypoint is unexercised.

## 10. Rebuild / change the model

`workflow_dispatch` the workflow, or change `HF_REPO` / `HF_REV` / `GGUF_FILE` /
`GGUF_SHA256` in the Dockerfile **and pick a new tag** -- never `latest`. Only
the Dockerfile and the workflow file trigger a build; `hymt2_prompt.py` and this
README are not part of the image, so editing them costs nothing.
