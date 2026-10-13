#!/usr/bin/env bash
# net_check.sh — 租機後網路體檢（開機 5 分鐘內跑，做任何 setup 之前）
# 最後一行輸出 NET_OK 或 NET_BAD
# 用法： scp 上去後  ssh ... "bash /root/net_check.sh" </dev/null
set -u

FAIL=0
SPEED_MIN=5000000   # 5 MB/s

echo "== DNS =="
for host in pypi.org huggingface.co github.com; do
  if getent hosts "$host" >/dev/null 2>&1; then
    echo "DNS $host ok"
  else
    echo "DNS $host FAIL"
    FAIL=1
  fi
done

echo "== HTTP =="
for url in "https://pypi.org/simple/pip/" "https://huggingface.co/" "https://github.com/"; do
  out=$(curl -sL -o /dev/null -m 15 -w "%{http_code} %{time_total}" "$url" 2>/dev/null || echo "000 0")
  code=${out%% *}
  echo "$url -> $out"
  case "$code" in
    2*|3*) ;;
    *) FAIL=1 ;;
  esac
done

echo "== SPEED (10MB via cloudflare) =="
speed=$(curl -sL -r 0-10485759 -m 40 -o /dev/null -w "%{speed_download}" \
  "https://speed.cloudflare.com/__down?bytes=10485760" 2>/dev/null || echo 0)
speed=${speed%%.*}
echo "download speed = ${speed} B/s (~$(( speed / 1024 / 1024 )) MB/s)"
if [ "${speed:-0}" -lt "$SPEED_MIN" ]; then
  echo "speed below 5MB/s -> FAIL"
  FAIL=1
fi

echo "== STABILITY (x3) =="
for i in 1 2 3; do
  curl -s -o /dev/null -m 10 -w "try$i %{http_code} %{time_total}s\n" https://huggingface.co/ || FAIL=1
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
