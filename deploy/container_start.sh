#!/usr/bin/env bash
# Container entrypoint: start OCR service + ollama, gate the model pull on HF speed.
# Everything else is already baked into the image, so this is fast and reliable.
set -u
export PATH=/usr/local/bin:/opt/conda/bin:$PATH
# Use the model weights baked into the image; never call HuggingFace at runtime
# (datacenter IPs are often unreachable/throttled, which made the first OCR
# request hang retrying the Hub and fail).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OLLAMA_HOST="0.0.0.0:8000"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-8192}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"

SAKURA_MODEL="${SAKURA_MODEL:-hf.co/SakuraLLM/Sakura-GalTransl-14B-v3.8:Q4_K_M}"
MODEL_URL="https://huggingface.co/SakuraLLM/Sakura-GalTransl-14B-v3.8/resolve/main/Sakura-Galtransl-14B-v3.8.gguf"

# HF speed probe -> /root/hf_mbps.txt (exposed by the OCR /health)
HF=$(curl -sL -r 0-52428799 -m 25 -o /dev/null -w '%{speed_download}' "$MODEL_URL" 2>/dev/null || echo 0)
HF=${HF:-0}
echo "$(( ${HF%%.*} / 1048576 ))" > /root/hf_mbps.txt
echo "HF_MBPS=$(cat /root/hf_mbps.txt)"

# OCR service first, so /health (with hf_mbps) is reachable during any pull
nohup env PORT=8080 python /root/ocr_server.py >/root/ocr.log 2>&1 </dev/null &

# ollama
nohup env OLLAMA_HOST="$OLLAMA_HOST" OLLAMA_CONTEXT_LENGTH="$OLLAMA_CONTEXT_LENGTH" \
  OLLAMA_KEEP_ALIVE="$OLLAMA_KEEP_ALIVE" ollama serve >/root/ollama.log 2>&1 </dev/null &
sleep 8

# Pull the model only if it is not already baked in AND HF is fast enough.
if OLLAMA_HOST=127.0.0.1:8000 ollama list 2>/dev/null | grep -qi "sakura"; then
  echo "sakura model already present (baked)"
elif [ "$(cat /root/hf_mbps.txt)" -ge 8 ]; then
  OLLAMA_HOST=127.0.0.1:8000 ollama pull "$SAKURA_MODEL" || echo "PULL_FAIL"
else
  echo "SKIP_PULL_SLOW_HF"
fi

echo "CONTAINER_READY"
wait
