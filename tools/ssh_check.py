"""Inspect a rented instance over SSH: processes, GPU, service logs, memory.

The image's OCR service logs to /root/ocr.log and serves it over HTTP as
`GET /logs` - but SSH is the only way to see everything at once (ollama's log,
nvidia-smi, free, the container start log). Read-only.

    python tools/ssh_check.py --instance 12345678
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from mth import vast as V  # noqa: E402

CMDS = [
    ("processes", "ps aux | grep -E 'ollama|ocr_server|llama-server' | grep -v grep"),
    ("gpu", "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader"),
    ("memory", "free -h"),
    ("ocr /health", "curl -s -m 5 http://127.0.0.1:8080/health"),
    ("llm /v1/models", "curl -s -m 8 http://127.0.0.1:8000/v1/models | head -c 200"),
    ("ocr.log tail", "tail -n 40 /root/ocr.log"),
    ("start.log", "tail -n 25 /root/start.log"),
    ("errors in ocr.log", "grep -iE 'uncaught|traceback|error' /root/ocr.log | tail -n 15"),
    ("dmesg oom", "dmesg 2>/dev/null | grep -iE 'oom|kill' | tail -n 10"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instance", type=int, required=True)
    args = ap.parse_args()

    key = os.environ.get("VAST_API_KEY")
    if not key:
        print("VAST_API_KEY is not set")
        return 2

    priv = V.ensure_ssh_key(key, REPO)
    targets = V.ssh_targets(key, args.instance)
    print("ssh targets: %s" % targets)
    target = V.wait_ssh(priv, targets, timeout=90)
    if not target:
        print("ssh unavailable (during the image pull the container is not up yet)")
        return 3
    print("ssh: %s" % (target,))

    for label, cmd in CMDS:
        rc, out, err = V._ssh(priv, target[0], target[1], cmd, timeout=90)
        print("\n===== %s =====" % label)
        print((out or "").strip()[:2000] or "(empty)")
        tail = "\n".join((err or "").strip().splitlines()[-2:])
        if tail and "Permanently added" not in tail:
            print("  [stderr] " + tail[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
