# 19 — Galgame translation images (side branches `sakura-ollama`, `hymt2-heretic`)

> Written 2026-11-18. Read this before touching either side branch, or before
> renting a machine for one of their images. These branches do **not** serve
> hanhua's manga pipeline: they build **translation-only** endpoints for the
> owner's separate **galgame** (visual-novel script) pipeline, which lives in a
> different project (its client is `translate_sakura.py`, with a `--profile`
> per model). Nothing on master depends on them, and master's tools do not fit
> them (section 6).

## 1. Branch map

| | `master` | `sakura-ollama` | `hymt2-heretic` |
|---|---|---|---|
| Purpose | manga archives (OCR + inpaint + translate) | galgame scripts, **Japanese** -> Chinese | galgame scripts incl. adult H-scenes, **English** -> Chinese |
| Branched from | — | `5bf9b91` (master, 2026-10) | `a94c8fb` (`sakura-ollama` tip) |
| Image tag (`<DOCKERUSR>/hanhua-gpu:`) | `<version>`, `latest` | `sakura-v3.8-q6k` | `hymt2-30b-a3b-heretic-q4km` |
| Built by | `build-image.yml`, **only on a `v*` tag on master** | `build-sakura-image.yml` | `build-hymt2-image.yml` |
| Build trigger | version tag | push to the branch touching `deploy/ollama-sakura/**` or its workflow | push to the branch touching `deploy/hymt2-heretic/Dockerfile` or its workflow |
| Server | ollama (8000) + OCR service (8080) | ollama, `:11434` | llama.cpp `llama-server`, `:11434` |
| Model | Sakura-14B-Qwen3-v1.5 Q6_K | Sakura-GalTransl-14B-v3.8 Q6_K (12.12 GB), model name `sakura-galtransl-14b` | Hy-MT2-30B-A3B-uncensored-heretic i1-Q4_K_M (18.24 GB), model name `hy-mt2-heretic` |
| Prompt | Qwen3 Sakura system prompt (`mth/providers.py`) | GalTransl v3.8's own system prompt (client profile `galtransl`) | **no system prompt**; Hy-MT2 templates, `deploy/hymt2-heretic/hymt2_prompt.py` |
| Booted on a GPU | yes (many batches) | not recorded in this repo | **no — never booted yet** |
| Docs | this folder | `deploy/ollama-sakura/README.md` | `deploy/hymt2-heretic/README.md` (the full usage guide) |

Each branch's workflow pushes **exactly one tag**, distinct from every other
tag in the repository, and never triggers on master. The side branches carry
master's tree as of their branch point; their copies of `llm_handoff/`,
`AGENTS.md` and `tools/` are therefore **stale** — master is the source of truth
for everything except their own `deploy/<image>/` folder and workflow.

## 2. `sakura-ollama` in brief

- Lean image: `ollama/ollama:latest` + the GGUF imported via a Modelfile, no OCR
  stack, boot makes no network calls.
- History (2026-10-15): the first build baked the Qwen3 Sakura under tag
  `sakura-qwen3-q6k` (wrong model for the galgame client); fixed to
  GalTransl-v3.8 in `86fd825`. `a94c8fb` added `delete-dockerhub-tag.yml`: a
  file under `ops/delete-tag/` whose content is a tag name, pushed to
  `sakura-ollama`, deletes that tag from Docker Hub (`ops/delete-tag/qwen3.txt`
  requested `sakura-qwen3-q6k`). Check Docker Hub before assuming a tag is gone.
- Client side: `translate_sakura.py --profile galtransl` in the galgame project.

## 3. `hymt2-heretic`: the decision record

### 3.1 The request

"A model of similar size to the sakura-ollama one, specialised for galgame
H-scenes: explicit vocabulary, dirty talk, onomatopoeia", on a new branch.
Direction is **English -> Chinese**. (The first pass wrongly assumed Japanese
because of the Sakura history; the owner corrected it. **Never infer the source
language from earlier models.**) The owner wants **one decisive pick**, not a
comparison table or an A/B plan.

### 3.2 The pick and why

**Hy-MT2-30B-A3B-uncensored-heretic, mradermacher imatrix Q4_K_M.**

- Base `tencent/Hy-MT2-30B-A3B` (2026-05): a dedicated translation model,
  EN<->ZH its strongest direction, MoE 30B total / 3B active (fast), Apache-2.0.
- Decensor: Heretic v1.4 (ARA) by OS-Software, refusals 100/100 -> 0/100, KL
  divergence 0.0276. Caveat: the refusal set was **Japanese**; English is
  expected, not measured.
- Quant: i1-Q4_K_M (18,236,703,488 bytes) is the largest that fits a 24 GB card
  with KV headroom; Q6_K (24.8 GB) / Q8_0 (32.1 GB) need 32-48 GB and a new tag.
- No open model is trained specifically on adult galgame text (searched
  2026-11-18). The fallback, only if this one measurably fails, is an
  abliterated ~27B chat model (better long context, several times slower, not a
  faithful line-aligned translator).

### 3.3 Image design

- Base `ghcr.io/ggml-org/llama.cpp:server-cuda-b12019` (CUDA 12.8; entrypoint
  `/app/llama-server`). Architecture `hy_v3` landed in llama.cpp b9993
  (PR #25395); b12019's `src/llama-arch.cpp` lists it. ollama support for
  `hy_v3` was unverified, hence llama.cpp (unlike `sakura-ollama`).
  (GHCR tag lists are paginated; the build tags were found by following the
  `Link` header.)
- Model downloaded at **build** time from a pinned HF commit
  (`862b7d66847b339c0594b8de39d38b035ee75f25`), hf-mirror fallback, verified by
  sha256 `b0358d0319aae3f821fef4989bd85f409336139bfc7db59fd049848b9882321c`.
- Config via `LLAMA_ARG_*` env: model `/models/model.gguf`, alias
  `hy-mt2-heretic`, `0.0.0.0:11434`, `-ngl 999`, ctx 32768 shared by 4 slots
  (8192 per request), `--jinja`, `LLAMA_ARG_UI=0` (the old `NO_WEBUI` name does
  not exist in b12019). CMD sets the model card's sampling: temp 0.7, top_p 1.0,
  top_k 0 (= off), repeat penalty 1.0. `HEALTHCHECK` re-pointed to `:11434/health`.
- VRAM: ~17.0 GiB weights + 3.0 GiB KV (48 layers x 4 KV heads x 128 x K,V x
  fp16 = 96 KiB/token x 32768) + buffers ~= 21 GiB.
- The GGUF chat template defaults to `reasoning_effort = no_think`.

### 3.4 Script policy: Simplified out, Traditional by OpenCC

Owner's rule: "the model side must not lose capability". So the prompt targets
**`简体中文`** (the model's strongest Chinese, and where its slang/SFX vocabulary
is), everything put into the prompt is Simplified (glossary targets and speaker
names go through OpenCC `tw2sp`), and the client converts the output with
OpenCC **`s2twp`** (Taiwan characters + Taiwan vocabulary, context-aware for
one-to-many characters: 干净->乾淨, 干吧->幹吧, 头发->頭髮 — tested). `s2t` / `s2tw` are
wrong (no Taiwan vocabulary). Spot-check 干/发/后/面 in the first real batch.

### 3.5 Prompt

`hymt2_prompt.py` (stdlib + OpenCC) is the reference, to be copied into the
galgame client as its `hymt2` profile: the card's **personalization** template
(one user message, no system message) with numbered requirements — house style
for adult scenes (`STYLE_ADULT`), only the glossary terms present in the line
(whole word, case-insensitive), speaker, the 3 previous lines marked "do not
translate", output-only, target `简体中文`. One script line per request, streamed,
retried; on a mechanical problem it retries once with the card's plain
template. CLI: `gate` (3 lines: plain, explicit, SFX; exit 1 on failure) and
`tr`. The README's sections 6-7 are the full spec and the client checklist.

### 3.6 Build record

| | |
|---|---|
| Commits | `c0e5e5f` (ci + Dockerfile), `8685102` (review fixes), `43a7337` (README + prompt), `53de592` (Simplified + s2twp), plus this handoff |
| Build | run `44242886554`, success, 18m13s, 2026-11-18 |
| Digest | `sha256:06bc4600bdded260c017758bfb4ca8f600ecc5f849f6b385082c9d33b23f975a` |
| Verified without a GPU | `hy_v3` in b12019; every env var name in b12019 `common/arg.cpp`; `top-k 0` = off; no_think default; base image has curl; VRAM arithmetic; offline mock-SSE test of the client (prompt shape, glossary matching, tw2sp/s2twp, streaming, retry on a dropped connection, fallback, gate) |
| **Not verified** | that the GGUF loads and serves on a GPU; translation quality; English refusal rate; the personalization prompt on real H-scene text |

## 4. CI behaviour learned here (applies to every branch)

- A workflow's `paths` filter is evaluated over the **whole push** (all commits
  pushed together), not the HEAD commit.
- `[skip ci]` is honoured only when it is in the **HEAD** commit's message.
- So a docs commit pushed together with an earlier workflow-file commit
  **triggers a build**. Push the build-relevant commit alone first, then the
  rest (`git push origin <sha>:<branch>`, then `git push origin <branch>`).
- `build-hymt2-image.yml` only watches the Dockerfile and itself: README /
  `hymt2_prompt.py` / `llm_handoff/` edits cost nothing.
- The runner has ~104 GB free after the standard cleanup list; no GHA cache
  (10 GB limit < model layer).

## 5. The first GPU run of `hymt2-heretic` (next session)

**Money rules first** (`deploy/VASTAI_PLAYBOOK.md`): report the budget and get
explicit consent before renting; $1 cap; one machine; a failed run keeps the
machine for debugging; verify teardown with `show instances`.

Budget basis: a 24 GB card (3090) is typically $0.15-0.20/hr; the image is
~21 GB (no host has it cached yet), so pull 7-25 min + load 1-3 min + gate
1 min ~= 0.5 h ~= **$0.10**. Report the actual offer before renting.

**Blocker to fix first, for free: `hanhua.py vast up` cannot launch this image
correctly as it stands** (master `mth/vast.py`):

| `vast up` today | This image needs |
|---|---|
| `create_instance(... ports=("8000","8080"))` | port **11434** only |
| offer filter `cuda_max_good >= 12.1` | **`>= 12.8`** (b12019 is CUDA 12.8) |
| `--ssh` launch: vast runs its own entrypoint, so the image's ENTRYPOINT/CMD **do not run**; `--baked` onstart runs `/root/container_start.sh` (master image only) | an onstart that starts `llama-server` itself (below) |
| readiness: ports 8000 + 8080 (`wait_http`, OCR `/health`) | `GET :11434/health` == 200, then `/v1/models` lists `hy-mt2-heretic` |

AGENTS.md says rental happens only through `hanhua.py vast up` and no other
script may rent. So the right fix is a small, offline-testable change to
`mth/vast.py` / `hanhua.py vast` on master: `--ports`, `--min-cuda`, an
`--onstart` override, and a readiness URL; `--dry-run` it, then ask for consent.
Do not hand-roll a `vastai create instance` call.

The onstart for this image (explicit flags, so it works whether or not vast
passes the image's `ENV` to the onstart shell):

```bash
nohup /app/llama-server -m /models/model.gguf --alias hy-mt2-heretic \
  --host 0.0.0.0 --port 11434 -ngl 999 -c 32768 -np 4 --jinja \
  --temp 0.7 --top-p 1.0 --top-k 0 --repeat-penalty 1.0 \
  >/root/llama.log 2>&1 </dev/null & echo started
```

(If `/app/llama-server` fails to find its shared libraries when started this
way, add `LD_LIBRARY_PATH=/app` — the image's entrypoint runs it from `/app`.)

Then, in order:
1. `deploy/net_check.sh` within 5 minutes of boot (playbook rule).
2. `curl :<mapped>/health` until 200 (503 while loading), `curl .../v1/models`.
3. `python deploy/hymt2-heretic/hymt2_prompt.py --api-base http://<host>:<mapped>/v1 gate`
   (needs `pip install opencc-python-reimplemented` locally).
4. If the owner provides lines: a ~50-line spot check of real H-scene text;
   check euphemism, refusals, one-to-many characters after `s2twp`.
5. Pull back the log (`/root/llama.log`) and results, destroy, confirm
   `show instances` is empty. Report money first.

Not applicable to this image: `tools/gate.py`, `tools/rent_and_run.py`'s
gate/run stages, ollama `/api/*` (`/api/tags`, `keep_alive: 0`), ports 8000/8080.

## 6. Open items

| # | Item | Needs |
|---|---|---|
| P0 | Make `hanhua.py vast up` able to boot a non-master image (section 5) | code change on master, tested with `--dry-run`, no spend |
| P0 | First GPU boot + gate of `hymt2-30b-a3b-heretic-q4km` | owner consent + budget report |
| P1 | `hymt2` profile in the galgame client (`translate_sakura.py`) per README section 7 | the owner gives that project's path |
| P1 | 50-line real H-scene spot check; tune `STYLE_ADULT` / glossary from it | owner-provided lines |
| P2 | If quality on Q4_K_M is the bottleneck: Q6_K on a 32 GB+ card, **new tag** | a measured reason |
