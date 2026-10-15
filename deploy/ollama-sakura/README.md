# ollama-sakura -- lean translation endpoint image

A minimal GPU image: the official `ollama/ollama` runtime plus the Sakura
translation model baked in. Boots make **no network calls** (HuggingFace is
unreachable from many rented hosts -- see `llm_handoff/12_hf_network_audit.md`).

| | |
|---|---|
| Built by | `.github/workflows/build-sakura-image.yml` (branch `sakura-ollama`) |
| Pushed to | `docker.io/<DOCKERUSR>/hanhua-gpu:sakura-v3.8-q6k` |
| Model | `sakura-galtransl-14b` = Sakura-GalTransl-14B-v3.8 Q6_K (12.12 GB), ctx 16384 |

The tag is deliberately distinct from the master-branch image (`:latest`,
`<version>`); the two workflows never push the same tag.
(An earlier build of this workflow produced the Qwen3-v1.5 variant under tag
`sakura-qwen3-q6k`; that tag can coexist or be deleted from Docker Hub.)

## Run

```bash
docker run --gpus all -p 11434:11434 <DOCKERUSR>/hanhua-gpu:sakura-v3.8-q6k
curl http://localhost:11434/v1/models
```

OpenAI-compatible API: `http://<host>:11434/v1`.
Use the model's own official system prompt (Sakura-GalTransl v3.8 ships its
own; for the galgame pipeline that is `translate_sakura.py --profile galtransl`).

## Rebuild / change the model

`workflow_dispatch` the workflow, or change the `GGUF_URL` build arg for
another quant/release **and pick a new tag** -- never `latest` (that tag
belongs to the master-branch image).