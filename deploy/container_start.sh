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
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-8192}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"

SAKURA_MODEL="${SAKURA_MODEL:-hf.co/SakuraLLM/Sakura-GalTransl-14B-v3.8:Q4_K_M}"
MODEL_URL="https://huggingface.co/SakuraLLM/Sakura-GalTransl-14B-v3.8/resolve/main/Sakura-Galtransl-14B-v3.8.gguf"

# OCR service first, so /health is reachable while everything else is starting.
# It runs under a restart loop: a single bad page must not take the whole service
# down, and GET /logs must stay reachable after a crash so the cause can be read.
nohup bash -c 'while true; do
    PORT=8080 python /root/ocr_server.py
    echo "[start] ocr_server exited rc=$? - restarting in 2s"
    sleep 2
  done' >>/root/ocr.log 2>&1 </dev/null &

# ollama
nohup env OLLAMA_HOST="$OLLAMA_HOST" OLLAMA_CONTEXT_LENGTH="$OLLAMA_CONTEXT_LENGTH" \
  OLLAMA_KEEP_ALIVE="$OLLAMA_KEEP_ALIVE" ollama serve >/root/ollama.log 2>&1 </dev/null &
sleep 8

# /health exposes hf_mbps for information only (readiness never gates on it).
echo 0 > /root/hf_mbps.txt

if OLLAMA_HOST=127.0.0.1:8000 ollama list 2>/dev/null | grep -qi "sakura"; then
  echo "sakura model already present (baked); skipping HF probe and pull"
else
  # Fallback (image built with BAKE_SAKURA=0): measure HF throughput and only
  # pull when it is fast enough, otherwise let the controller fail fast and
  # switch machines instead of stalling on a bad-network host.
  HF=$(curl -sL -r 0-52428799 -m 25 -o /dev/null -w '%{speed_download}' "$MODEL_URL" 2>/dev/null || echo 0)
  HF=${HF:-0}
  echo "$(( ${HF%%.*} / 1048576 ))" > /root/hf_mbps.txt
  echo "HF_MBPS=$(cat /root/hf_mbps.txt)"
  if [ "$(cat /root/hf_mbps.txt)" -ge 8 ]; then
    OLLAMA_HOST=127.0.0.1:8000 ollama pull "$SAKURA_MODEL" || echo "PULL_FAIL"
  else
    echo "SKIP_PULL_SLOW_HF"
  fi
fi

echo "CONTAINER_READY"
wait
