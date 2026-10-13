#!/usr/bin/env bash
set -euo pipefail

# Sakura translation server for vast.ai (OpenAI-compatible via llama.cpp)
# usage: API_KEY=secret bash start_sakura.sh
#        MODEL=SakuraLLM/Sakura-GalTransl-7B-v3.7:Q6_K PORT=8000 bash start_sakura.sh

MODEL="${MODEL:-SakuraLLM/Sakura-GalTransl-14B-v3.8:Q4_K_M}"
PORT="${PORT:-8000}"
CTX="${CTX:-8192}"
PARALLEL="${PARALLEL:-4}"
NGL="${NGL:-99}"
API_KEY="${API_KEY:-}"

if ! command -v llama-server >/dev/null 2>&1; then
  echo "llama-server not found. Installing llama.cpp..."
  curl -LsSf https://llama.app/install.sh | sh || true
  export PATH="$HOME/.local/bin:$PATH"
fi

# -ngl 99  = offload all layers to GPU (RTX 4090 24GB)
# -np 4    = 4 parallel slots; per-slot context = CTX / np
ARGS=(--host 0.0.0.0 --port "$PORT" -c "$CTX" -np "$PARALLEL" -ngl "$NGL" --jinja)

if [ -e "$MODEL" ]; then
  ARGS+=(-m "$MODEL")
else
  ARGS+=(-hf "$MODEL")
fi

if [ -n "$API_KEY" ]; then
  ARGS+=(--api-key "$API_KEY")
fi

echo "starting Sakura on 0.0.0.0:${PORT}  model=${MODEL}"
echo "OpenAI endpoint: http://<vast-ip>:<mapped-port>/v1"
exec llama-server "${ARGS[@]}"
