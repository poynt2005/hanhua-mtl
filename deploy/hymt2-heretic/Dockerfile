# Lean llama.cpp server with an uncensored EN->ZH translation model baked in,
# for adult (H-scene) galgame text: explicit vocabulary, onomatopoeia, moaning.
#
# Model: Hy-MT2-30B-A3B-uncensored-heretic, imatrix Q4_K_M (18.24 GB)
#   base     : tencent/Hy-MT2-30B-A3B (translation MoE, 30B total / 3B active,
#              Apache-2.0, English->Chinese is its strongest direction)
#   decensor : OS-Software, Heretic v1.4 ARA; refusals 100/100 -> 0/100,
#              KL divergence 0.0276 vs the original
#   quant    : mradermacher i1-Q4_K_M (marked "recommended" on the repo)
#
# Why llama.cpp and not ollama (unlike the sakura-ollama image):
#   the architecture is `hy_v3` (HYV3ForCausalLM). llama.cpp supports it since
#   b9993 (PR #25395) and the pinned b12019 lists it in src/llama-arch.cpp;
#   ollama's support for it is unverified. llama-server also reads the GGUF file
#   directly, so the layer holds the model once (no separate ollama blob).
#
# VRAM (24 GB card): weights 18.24 GB (~17.0 GiB) + KV cache 96 KiB/token
#   (48 layers x 4 KV heads x 128 dim x K,V x fp16) x 32768 tokens = 3.0 GiB
#   + compute buffers ~= 21 GiB.
#
# The GGUF's chat template defaults to reasoning_effort = no_think, so replies
# carry no thinking block.
#
# Boot makes NO network calls (HuggingFace is unreachable from many rented
# hosts -- see llm_handoff/12_hf_network_audit.md).
#
# Built by .github/workflows/build-hymt2-image.yml (branch hymt2-heretic) and
# pushed as <DOCKERUSR>/hanhua-gpu:hymt2-30b-a3b-heretic-q4km. The tag is
# deliberately distinct from every other tag in that repository.
#
# Run (same port as the sakura-ollama image, so a client only changes the model):
#   docker run --gpus all -p 11434:11434 <DOCKERUSR>/hanhua-gpu:hymt2-30b-a3b-heretic-q4km
#   -> OpenAI-compatible API at http://<host>:11434/v1, model "hy-mt2-heretic"

# llama.cpp b12019 (2026-10-31): after hy_v3 support landed, >2 weeks old.
FROM ghcr.io/ggml-org/llama.cpp:server-cuda-b12019

# Pinned to a repo commit and verified by sha256, so a re-upload can never
# change the model silently.
ARG HF_REPO=mradermacher/Hy-MT2-30B-A3B-uncensored-heretic-i1-GGUF
ARG HF_REV=862b7d66847b339c0594b8de39d38b035ee75f25
ARG GGUF_FILE=Hy-MT2-30B-A3B-uncensored-heretic.i1-Q4_K_M.gguf
ARG GGUF_SHA256=b0358d0319aae3f821fef4989bd85f409336139bfc7db59fd049848b9882321c

RUN set -eux; \
    if ! command -v curl >/dev/null 2>&1; then \
        apt-get update; \
        apt-get install -y --no-install-recommends curl ca-certificates; \
        rm -rf /var/lib/apt/lists/*; \
    fi; \
    mkdir -p /models; \
    ( curl -fL --retry 5 --retry-delay 5 -o /models/model.gguf \
          "https://huggingface.co/${HF_REPO}/resolve/${HF_REV}/${GGUF_FILE}" \
      || curl -fL --retry 5 --retry-delay 5 -o /models/model.gguf \
          "https://hf-mirror.com/${HF_REPO}/resolve/${HF_REV}/${GGUF_FILE}" ); \
    echo "${GGUF_SHA256}  /models/model.gguf" | sha256sum -c -

# llama-server reads LLAMA_ARG_* from the environment; override any of these
# with `docker run -e`. Context is shared by the slots: 32768 / 4 = 8192 tokens
# per request, which is the model's served context and ample for a script chunk.
ENV LLAMA_ARG_MODEL=/models/model.gguf \
    LLAMA_ARG_ALIAS=hy-mt2-heretic \
    LLAMA_ARG_HOST=0.0.0.0 \
    LLAMA_ARG_PORT=11434 \
    LLAMA_ARG_N_GPU_LAYERS=999 \
    LLAMA_ARG_CTX_SIZE=32768 \
    LLAMA_ARG_N_PARALLEL=4 \
    LLAMA_ARG_JINJA=1 \
    LLAMA_ARG_UI=0

# The base image's healthcheck probes :8080; the server listens on 11434 here.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fs http://localhost:11434/health || exit 1

EXPOSE 11434

# Server-side sampling defaults = the model card's recommendation for 30B-A3B
# (temperature 0.7, top_p 1.0, top_k off, repetition penalty 1.0). A client's
# request parameters still take precedence.
CMD ["--temp", "0.7", "--top-p", "1.0", "--top-k", "0", "--repeat-penalty", "1.0"]
