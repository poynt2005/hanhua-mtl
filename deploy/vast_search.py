import argparse
import json
import os
import urllib.error
import urllib.request

ENDPOINT = "https://console.vast.ai/api/v0/bundles/"

SWEET_SPOT = ["RTX 3090", "RTX 4090", "RTX A5000", "RTX 4080", "RTX 4060 Ti", "L4"]


def search(api_key, query):
    data = json.dumps(query).encode("utf-8")
    req = urllib.request.Request(ENDPOINT, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + api_key)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        raise SystemExit("HTTP %s: %s" % (err.code, err.read().decode("utf-8", "replace")[:300]))
    except urllib.error.URLError as err:
        raise SystemExit("network error: %s" % err)


def main():
    parser = argparse.ArgumentParser(description="find cost-effective vast.ai GPU offers")
    parser.add_argument("--key", default=os.environ.get("VAST_API_KEY", ""))
    parser.add_argument("--gpu", default="", help="e.g. 'RTX 3090'; empty = cost-effective list")
    parser.add_argument("--min-vram", type=int, default=24, help="GB")
    parser.add_argument("--min-down", type=int, default=200, help="min inet_down Mbps (0=off)")
    parser.add_argument("--min-up", type=int, default=50, help="min inet_up Mbps (0=off)")
    parser.add_argument("--order", default="value", choices=["value", "cheap"],
                        help="value=best dlperf/$, cheap=lowest $/hr")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    if not args.key:
        raise SystemExit("set VAST_API_KEY or pass --key")

    if args.order == "value":
        order = [["dlperf_per_dphtotal", "desc"]]
    else:
        order = [["dph_total", "asc"]]

    query = {
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "num_gpus": {"eq": 1},
        "gpu_ram": {"gte": args.min_vram * 1024},
        "direct_port_count": {"gte": 2},
        "cuda_max_good": {"gte": 12.1},
        "order": order,
        "type": "on-demand",
        "limit": args.limit,
    }
    if args.gpu:
        query["gpu_name"] = {"eq": args.gpu}
    if args.min_down:
        query["inet_down"] = {"gte": args.min_down}
    if args.min_up:
        query["inet_up"] = {"gte": args.min_up}

    data = search(args.key, query)
    offers = data.get("offers", [])

    if not args.gpu:
        print("=== cost-effective candidates (VRAM>=%dGB) ===" % args.min_vram)
    else:
        print("=== %s (VRAM>=%dGB) ===" % (args.gpu, args.min_vram))
    print("%-9s %-12s %6s %8s %7s %6s %-22s %s" %
          ("id", "gpu", "vram", "$/hr", "dlperf/", "down", "geolocation", "reliability"))
    for offer in offers:
        print("%-9s %-12s %5dG %8.3f %7.1f %6.0f %-22s %.3f" % (
            offer.get("id"),
            offer.get("gpu_name"),
            int(offer.get("gpu_ram", 0) / 1024),
            offer.get("dph_total", 0.0),
            offer.get("dlperf_per_dphtotal", 0.0),
            offer.get("inet_down", 0.0),
            (offer.get("geolocation") or "")[:22],
            offer.get("reliability2", 0.0),
        ))
    print("\n提示：把 id 記下，用 vastai create instance <id> 建立，並開 8000/8080 埠。")
    print("     單價之外，請同時看 vram 是否 >=24G、down 速度（要抓模型）。")


if __name__ == "__main__":
    main()
