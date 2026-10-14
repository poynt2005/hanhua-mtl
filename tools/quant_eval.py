"""Evaluate the cost of a stronger model quantisation, from real data.

Fetches the actual GGUF file sizes from HuggingFace and the cheapest live
vast.ai price per GPU tier (query only - it never rents), then prints the
cost-per-volume implications.

This is how the Q6_K decision was made and how the next one should be:
see llm_handoff/13_operating_manual.md for the reasoning it supports.

    python tools/quant_eval.py --repo SakuraLLM/Sakura-Transl-14B-v3.8
    python tools/quant_eval.py --repo SakuraLLM/Sakura-14B-Qwen3-v1.5-GGUF --gpus "RTX 3090,RTX A6000,A100 PCIE"
"""
import argparse
import json
import os
import sys
import urllib.request

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
from mth import vast as V  # noqa: E402

# Measured on this project's own runs (see llm_handoff/15_run_log.md):
SECONDS_PER_PAGE_AT_Q4 = 15.0      # whole pipeline, Q4_K_M, 8192 ctx
COST_PER_HOUR = 0.153              # the V100 that ran the batch


def hf_gguf_sizes(repo):
    url = "https://huggingface.co/api/models/%s?blobs=true" % repo
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return sorted(((s["rfilename"], s.get("size") or 0)
                   for s in data.get("siblings", [])
                   if s["rfilename"].lower().endswith(".gguf")), key=lambda x: x[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="HuggingFace repo id")
    ap.add_argument("--gpus", default="RTX 3090,RTX A6000,A100 PCIE,L40S",
                    help="comma-separated vast.ai gpu names")
    ap.add_argument("--pages", type=int, default=250, help="pages per volume")
    ap.add_argument("--max-price", type=float, default=3.0)
    args = ap.parse_args()

    print("=== %s: GGUF sizes ===" % args.repo)
    try:
        files = hf_gguf_sizes(args.repo)
    except Exception as exc:  # noqa: BLE001
        print("  could not list files: %s" % exc)
        files = []
    for name, size in files:
        print("  %-56s %6.2f GB" % (name, size / 1e9))
    if not files:
        print("  (check the repo id)")

    print("\n=== cheapest live vast.ai offer per tier (query only, nothing rented) ===")
    key = os.environ.get("VAST_API_KEY")
    if not key:
        print("  VAST_API_KEY not set - skipping price lookup")
        return 1
    for gpu in [g.strip() for g in args.gpus.split(",") if g.strip()]:
        try:
            offers = V.search_offers(key, min_vram=24, min_down=100, min_up=50,
                                     max_price=args.max_price, gpu=gpu, order="cheap", limit=5)
        except Exception as exc:  # noqa: BLE001
            print("  %-14s query failed: %s" % (gpu, exc))
            continue
        if not offers:
            print("  %-14s no offers at <= $%.2f/hr" % (gpu, args.max_price))
            continue
        best = offers[0]
        hours = SECONDS_PER_PAGE_AT_Q4 * args.pages / 3600.0
        print("  %-14s $%.3f/hr  %3.0f GB VRAM  -> %.0f min/volume = $%.2f/volume"
              % (gpu, best.get("dph_total", 0), (best.get("gpu_ram") or 0) / 1024.0,
                 hours * 60, hours * best.get("dph_total", 0)))
    print("\n(baseline on the machine actually used: $%.3f/hr -> $%.2f per %d-page volume)"
          % (COST_PER_HOUR, COST_PER_HOUR * SECONDS_PER_PAGE_AT_Q4 * args.pages / 3600.0,
             args.pages))
    return 0


if __name__ == "__main__":
    sys.exit(main())
