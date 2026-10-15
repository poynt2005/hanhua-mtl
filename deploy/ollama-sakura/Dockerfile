# Lean ollama runtime with the Sakura translation model baked in.
#
# Purpose: a translation endpoint whose boot makes NO network calls at all --
# HuggingFace is unreachable from many rented GPU hosts (see
# llm_handoff/12_hf_network_audit.md), so the model is baked at build time.
#
# Built by .github/workflows/build-sakura-image.yml (branch sakura-ollama) and
# pushed to Docker Hub as <DOCKERUSR>/hanhua-gpu:sakura-qwen3-q6k. The tag is
# deliberately distinct from the master-branch image tags (:latest, <version>).
#
# Run:
#   docker run --gpus all -p 11434:11434 <DOCKERUSR>/hanhua-gpu:sakura-qwen3-q6k
#   -> OpenAI-compatible API at http://<host>:11434/v1

FROM ollama/ollama:latest

ARG GGUF_URL=https://huggingface.co/SakuraLLM/Sakura-14B-Qwen3-v1.5-GGUF/resolve/main/sakura-14b-qwen3-v1.5-q6k.gguf
ARG GGUF_MIRROR=https://hf-mirror.com/SakuraLLM/Sakura-14B-Qwen3-v1.5-GGUF/resolve/main/sakura-14b-qwen3-v1.5-q6k.gguf
ARG GGUF_MIN_BYTES=12000000000
ARG MODEL=sakura-qwen3-q6k
ARG NUM_CTX=16384

# One layer: ensure curl, download the GGUF (mirror fallback), import it as an
# ollama model, drop the raw GGUF. The GGUF is created and deleted inside the
# same RUN, so the layer keeps only the resulting ollama blob.
RUN set -eux; \
    if ! command -v curl >/dev/null 2>&1; then \
        apt-get update; \
        apt-get install -y --no-install-recommends curl ca-certificates; \
        rm -rf /var/lib/apt/lists/*; \
    fi; \
    ( curl -fL --retry 5 --retry-delay 5 -o /root/model.gguf "$GGUF_URL" \
      || curl -fL --retry 5 --retry-delay 5 -o /root/model.gguf "$GGUF_MIRROR" ); \
    sz=$(stat -c%s /root/model.gguf); echo "GGUF bytes=$sz"; \
    [ "$sz" -ge "$GGUF_MIN_BYTES" ] || { echo "GGUF download incomplete"; exit 1; }; \
    printf 'FROM /root/model.gguf\nPARAMETER num_ctx %s\n' "$NUM_CTX" > /root/Modelfile; \
    ollama serve & \
    pid=$!; \
    for i in $(seq 1 90); do ollama list >/dev/null 2>&1 && break; sleep 2; done; \
    ollama create "$MODEL" -f /root/Modelfile; \
    ollama list; \
    kill "$pid"; \
    rm -f /root/model.gguf

ENV OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_KEEP_ALIVE=-1 \
    OLLAMA_NUM_PARALLEL=4 \
    OLLAMA_FLASH_ATTENTION=1 \
    OLLAMA_KV_CACHE_TYPE=q8_0

EXPOSE 11434