"""Prove (or disprove) that the endpoint drops NON-STREAMING requests.

This is the experiment that found the root cause of every aborted batch: a
non-streaming completion is silent for the whole generation, and the GPU host's
port forward drops an idle socket after ~15 s. Measured at the time:
**4/6 non-streaming requests survived, 6/6 streaming ones did.**

Run it against a live endpoint to check a host, or to re-confirm the design
decision. It is read-only (a few short completions) and costs nothing but the
machine time you already pay for.

    python tools/stream_check.py --config config.json [--rounds 6] [--lines 10]
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mth import config as mthconfig  # noqa: E402
from mth import providers  # noqa: E402


def call(cfg, lines, stream, timeout=600):
    payload = {
        "model": cfg["text_model"],
        "messages": providers._sakura_messages(lines),
        "temperature": 0.3, "top_p": 0.8, "max_tokens": 4096, "stream": bool(stream),
    }
    req = urllib.request.Request(cfg["api_base"].rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + (cfg.get("api_key") or "ollama"))
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if stream:
                chunks = 0
                for raw in resp:
                    if raw.decode("utf-8", "replace").strip().startswith("data:"):
                        chunks += 1
                ok = chunks > 0
            else:
                ok = bool(json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"])
        return ok, time.time() - t0, None
    except Exception as exc:  # noqa: BLE001
        return False, time.time() - t0, type(exc).__name__


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--lines", type=int, default=10)
    args = ap.parse_args()

    cfg = mthconfig.load_config(args.config)
    print("endpoint: %s   model: %s" % (cfg.get("api_base"), cfg.get("text_model")))
    lines = ["テスト%d。%s" % (i, "長い行" * 20) for i in range(args.lines)]

    results = {}
    for mode in (False, True):
        label = "STREAMING" if mode else "NON-streaming"
        ok = 0
        for i in range(args.rounds):
            good, secs, err = call(cfg, lines, mode)
            ok += good
            print("  %-14s %d/%d  %5.1fs  %s" % (label, i + 1, args.rounds, secs, err or "ok"))
        results[label] = (ok, args.rounds)

    print("\nsummary")
    for label, (ok, total) in results.items():
        print("  %-14s %d/%d survived" % (label, ok, total))
    if results["STREAMING"][0] > results["NON-streaming"][0]:
        print("\n=> streaming is more reliable here, which is why the client streams.")
        return 0
    print("\n=> no difference observed this time (the drop is intermittent).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
