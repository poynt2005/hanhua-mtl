#!/usr/bin/env bash
# safe_destroy.sh — 銷毀 vast.ai instance 並「驗證」真的關了、停費了
# 用法： bash safe_destroy.sh <instance_id>
# 注意：銷毀前請先確認產物已 scp 回本機並驗過（見 R5/18、V31）。
set -euo pipefail

ID="${1:?usage: safe_destroy.sh <instance_id>}"

if ! command -v vastai >/dev/null 2>&1; then
  echo "vastai CLI not found. pip install vastai" >&2
  exit 1
fi

echo "== before =="
vastai show instances || true

# 新版 CLI 會跳 [y/N] 確認；非互動必須先餵 y（V25）
echo y | vastai destroy instance "$ID"

echo "== after (should be empty) =="
vastai show instances || true

echo "== account =="
vastai show user 2>/dev/null | grep -Ei "credit|total_spend" || true

echo
echo "請人工確認上面 'after' 為空、且沒有殘留 instance 在計費。"
