"""Pre-flight gate: run this BEFORE any batch, right after renting.

It checks the three things that have actually wasted money here:

  1. the OCR service is healthy AND its neural detector really loaded;
  2. the LLM answers at all;
  3. a 3-line translation comes back as exactly 3 non-empty Chinese lines with
     no reasoning/template noise - this catches a wrong chat template and
     reasoning models that emit ` thinking` into every answer, either of which
     would silently ruin a whole volume.

Exits non-zero on any failure and prints the raw reply for diagnosis.

    python tools/gate.py --config config.json
"""
import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mth import config as mthconfig  # noqa: E402
from mth import providers  # noqa: E402

LINES = ["おはよう。", "今日はいい天気だね。", "彼女は静かに微笑んだ。"]
POISON = (" thinking", "</think>", "<|im_start|>", "<|im_end|>", "思考", "分析")


def has_cjk(text):
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()

    cfg = mthconfig.load_config(args.config)
    print("model  : %s" % cfg.get("text_model"))
    print("api    : %s" % cfg.get("api_base"))
    print("ocr    : %s" % cfg.get("ocr_url"))
    print("stream : %s   retries: %s" % (cfg.get("stream"), cfg.get("retries")))

    fails = []

    # 1) OCR health -----------------------------------------------------------
    base = (cfg.get("ocr_url") or "").replace("/prepare", "")
    if base:
        try:
            health = json.loads(urllib.request.urlopen(base + "/health", timeout=15).read())
            print("\n[1] OCR /health: %s" % health)
            if not health.get("ok"):
                fails.append("OCR /health not ok")
            if not health.get("detector"):
                fails.append("OCR detector NOT loaded (detection would fall back to EasyOCR)")
        except Exception as exc:  # noqa: BLE001
            print("\n[1] OCR /health FAILED: %s" % exc)
            fails.append("OCR /health unreachable")
    else:
        fails.append("no ocr_url in the config")

    # 2) the LLM must turn 3 lines into 3 Chinese lines -----------------------
    reply = ""
    try:
        client = providers.build_client(cfg)
        reply = client.chat(cfg["text_model"], providers._sakura_messages(LINES),
                            temperature=0.3, top_p=0.8, max_tokens=512)
    except Exception as exc:  # noqa: BLE001
        print("\n[2] translation FAILED: %s" % exc)
        fails.append("translation request failed")

    if reply:
        got = reply.strip().split("\n")
        print("\n[2] raw reply (%d lines):" % len(got))
        for line in got:
            print("      %s" % line)
        if len(got) != len(LINES):
            fails.append("expected %d lines, got %d" % (len(LINES), len(got)))
        if not all(has_cjk(line) for line in got if line.strip()):
            fails.append("a translated line has no Chinese characters")
        noise = [p for p in POISON if p in reply]
        if noise:
            fails.append("reply contains reasoning/template noise: %s" % noise)

    print("\n" + "=" * 56)
    if fails:
        print("GATE FAILED:")
        for f in fails:
            print("  - %s" % f)
        print("Do NOT start a batch. See llm_handoff/14_thrashing_postmortem.md")
        return 1
    print("GATE OK - the model translates correctly and the detector is loaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
