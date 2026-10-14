"""Hammer the translation endpoint and report intermittent failures.

A single `ping` is not enough: this endpoint dropped roughly 1 in 5
non-streaming requests, and one dropped request used to abort a whole volume.
Run this against a live endpoint to see whether it is flaky right now.

    python tools/llm_check.py --config config.json [--rounds 8] [--lines 10]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mth import config as mthconfig  # noqa: E402
from mth import providers  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--lines", type=int, default=10)
    args = ap.parse_args()

    cfg = mthconfig.load_config(args.config)
    client = providers.build_client(cfg)
    print("endpoint: %s   model: %s   stream=%s retries=%s"
          % (cfg.get("api_base"), cfg.get("text_model"), cfg.get("stream"), cfg.get("retries")))

    lines = ["これは%d番目のテスト文です。今日はいい天気ですね。" % i for i in range(args.lines)]
    ok = 0
    for i in range(args.rounds):
        t0 = time.time()
        try:
            reply = client.chat(cfg["text_model"], providers._sakura_messages(lines),
                                temperature=0.3, top_p=0.8, max_tokens=4096)
            got = len(reply.strip().split("\n"))
            ok += 1
            print("  %d/%d  OK   %5.1fs  lines_in=%d lines_out=%d"
                  % (i + 1, args.rounds, time.time() - t0, len(lines), got))
        except Exception as exc:  # noqa: BLE001
            print("  %d/%d  FAIL %5.1fs  %s: %s"
                  % (i + 1, args.rounds, time.time() - t0, type(exc).__name__, exc))
    print("\n%d/%d succeeded" % (ok, args.rounds))
    print("(the client already streams and retries; a failure here means the "
          "endpoint itself is unhealthy)")
    return 0 if ok == args.rounds else 1


if __name__ == "__main__":
    sys.exit(main())
