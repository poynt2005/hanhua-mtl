#!/usr/bin/env bash
# net_check.sh — rented-box network health check (run within 5 min of boot).
# Only the sources we actually need (PyPI for pip, HuggingFace for models) are
# REQUIRED. GitHub is informational only: vast datacenter IPs are often
# rate-limited (403) there, which must NOT fail the check.
# Last line: NET_OK or NET_BAD
set -u

FAIL=0
SPEED_MIN=5000000   # 5 MB/s

echo "== DNS (required) =="
for host in pypi.org huggingface.co; do
  if getent hosts "$host" >/dev/null 2>&1; then
    echo "DNS $host ok"
  else
    echo "DNS $host FAIL"
    FAIL=1
  fi
done

echo "== HTTP (required) =="
for url in "https://pypi.org/simple/pip/" "https://huggingface.co/"; do
  out=$(curl -sL -o /dev/null -m 15 -w "%{http_code} %{time_total}" "$url" 2>/dev/null || echo "000 0")
  code=${out%% *}
  echo "$url -> $out"
  case "$code" in
    2*|3*) ;;
    *) FAIL=1 ;;
  esac
done

echo "== HTTP (informational, ignored) =="
curl -sL -o /dev/null -m 15 -w "github.com -> %{http_code} %{time_total}s\n" https://github.com/ 2>/dev/null || echo "github.com -> unreachable"

echo "== SPEED (max of sources; need >= 5MB/s) =="
best=0
for url in \
  "https://speed.cloudflare.com/__down?bytes=10485760" \
  "https://huggingface.co/SakuraLLM/Sakura-GalTransl-14B-v3.8/resolve/main/Sakura-Galtransl-14B-v3.8.gguf"; do
  s=$(curl -sL -r 0-10485759 -m 40 -o /dev/null -w "%{speed_download}" "$url" 2>/dev/null || echo 0)
  s=${s%%.*}
  echo "  ${url:0:60} -> ${s} B/s"
  [ "${s:-0}" -gt "$best" ] && best=$s
done
echo "best = $best B/s (~$(( best / 1024 / 1024 )) MB/s)"
if [ "$best" -lt "$SPEED_MIN" ]; then
  echo "speed below 5MB/s -> FAIL"
  FAIL=1
fi

echo "== STABILITY =="
for i in 1 2 3; do
  curl -s -o /dev/null -m 10 -w "try$i %{http_code} %{time_total}s\n" https://huggingface.co/ || true
done

echo "== GPU =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
  echo "nvidia-smi not found"
fi

if [ "$FAIL" -eq 0 ]; then
  echo NET_OK
else
  echo NET_BAD
fi
