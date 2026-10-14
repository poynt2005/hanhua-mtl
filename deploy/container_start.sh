#!/usr/bin/env bash
# Container entrypoint: bring up the OCR service and ollama.
#
# Every model weight is baked into the image, so a normal (BAKE_SAKURA=1) boot
# makes NO network calls at all: no HuggingFace, no GitHub. The HuggingFace speed
# probe and the model pull below only run in the fallback case where the Sakura
# GGUF was not baked in.
set -u
export PATH=/usr/local/bin:/opt/conda/bin:$PATH
# Load the baked weights; never phone HuggingFace at runtime. transformers checks
# the Hub on every model load even when the weights are already cached, and on
# hosts where the Hub is unreachable that check hangs until the request fails.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OLLAMA_HOST="0.0.0.0:8000"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-16384}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"

SAKURA_MODEL="${SAKURA_MODEL:-sakura-qwen3-q6k}"
# Sakura-14B-Qwen3-v1.5 Q6_K (12.12 GB). Imported with a Modelfile rather than
# `ollama pull :Q6_K`, which cannot resolve for this repo either.
MODEL_URL="${SAKURA_MODEL_URL:-https://huggingface.co/SakuraLLM/Sakura-14B-Qwen3-v1.5-GGUF/resolve/main/sakura-14b-qwen3-v1.5-q6k.gguf}"

# OCR service first, so /health is reachable while everything else is starting.
# It runs under a restart loop: a single bad page must not take the whole service
# down, and GET /logs must stay reachable after a crash so the cause can be read.
# NOTE: stdout is deliberately NOT redirected into /root/ocr.log - the server
# already appends every line to that file itself, and doing both duplicated every
# line (which made log-based progress counting unreliable).
nohup bash -c 'while true; do
    PORT=8080 python /root/ocr_server.py
    echo "[start] ocr_server exited rc=$? - restarting in 2s"
    sleep 2
  done' </dev/null &

# ollama
nohup env OLLAMA_HOST="$OLLAMA_HOST" OLLAMA_CONTEXT_LENGTH="$OLLAMA_CONTEXT_LENGTH" \
  OLLAMA_KEEP_ALIVE="$OLLAMA_KEEP_ALIVE" ollama serve >/root/ollama.log 2>&1 </dev/null &
sleep 8

# /health exposes hf_mbps for information only (readiness never gates on it).
echo 0 > /root/hf_mbps.txt

if OLLAMA_HOST=127.0.0.1:8000 ollama list 2>/dev/null | grep -qi "${SAKURA_MODEL%%:*}"; then
  echo "model ${SAKURA_MODEL} already present (baked); skipping HF probe and download"
else
  # Fallback (image built with BAKE_SAKURA=0): measure HF throughput and only
  # download when it is fast enough, otherwise let the controller fail fast and
  # switch machines instead of stalling on a bad-network host.
  HF=$(curl -sL -r 0-52428799 -m 25 -o /dev/null -w '%{speed_download}' "$MODEL_URL" 2>/dev/null || echo 0)
  HF=${HF:-0}
  echo "$(( ${HF%%.*} / 1048576 ))" > /root/hf_mbps.txt
  echo "HF_MBPS=$(cat /root/hf_mbps.txt)"
  if [ "$(cat /root/hf_mbps.txt)" -ge 8 ]; then
    curl -fL --retry 5 -o "/root/${SAKURA_MODEL}.gguf" "$MODEL_URL" \
      && printf 'FROM /root/%s.gguf\n' "$SAKURA_MODEL" > /root/Modelfile \
      && OLLAMA_HOST=127.0.0.1:8000 ollama create "$SAKURA_MODEL" -f /root/Modelfile \
      && rm -f "/root/${SAKURA_MODEL}.gguf" || echo "PULL_FAIL"
  else
    echo "SKIP_PULL_SLOW_HF"
  fi
fi

echo "CONTAINER_READY"
wait
