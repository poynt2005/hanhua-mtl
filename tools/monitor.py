"""Sample GPU and network counters on a rented instance, without SSH.

The renter-facing API has no per-instance traffic counter: the "Mbps" badge in
the web console is the offer's advertised `inet_up`/`inet_down`, not a
measurement (verified against the console bundle). What does exist is
`vastai execute`, which runs a command through the API and returns its output
from a URL - so the numbers can be read straight from the container.

Every sample is appended to a JSONL file, which is what makes a cost post-mortem
possible: pages/hour, seconds spent with the GPU idle, and whether the link -
not the model - was the bottleneck.

Read-only. It never starts, stops or destroys anything.

  python tools/monitor.py --instance 12345678 --interval 15 --out monitor.jsonl
"""
import argparse
import json
import os
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

COMMAND = (
    "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw,memory.used"
    " --format=csv,noheader,nounits; "
    "cat /proc/net/dev"
)


def _api(key, method, path, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        "https://console.vast.ai" + path, data=data, method=method)
    req.add_header("Authorization", "Bearer " + key)
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_command(key, iid, command, wait=25):
    """Queue one command and return its output (empty string if it is not ready)."""
    reply = _api(key, "PUT", "/api/v0/instances/command/%s" % iid, {"command": command})
    url = reply.get("result_url")
    if not url:
        return ""
    deadline = time.time() + wait
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                return resp.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 - the object appears a few seconds later
            time.sleep(2)
    return ""


def parse(text):
    """Split the combined output into a GPU dict and per-interface byte totals."""
    gpu = {}
    net = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line and "|" not in line:
            name, _, rest = line.partition(":")
            name = name.strip()
            if name in ("lo", "Inter-|", "face"):
                continue
            fields = rest.split()
            if len(fields) >= 9:
                net[name] = {"rx_bytes": int(fields[0]), "tx_bytes": int(fields[8])}
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            try:
                gpu = {
                    "gpu_util_pct": float(parts[0]),
                    "mem_util_pct": float(parts[1]),
                    "power_w": None if parts[2].lower().startswith("[") else float(parts[2]),
                    "vram_used_mb": float(parts[3]),
                }
            except ValueError:
                pass
    return gpu, net


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--interval", type=float, default=15.0, help="seconds between samples")
    ap.add_argument("--out", default="monitor.jsonl")
    ap.add_argument("--duration", type=float, default=0,
                    help="stop after N minutes (0 = until interrupted)")
    args = ap.parse_args()

    key = os.environ.get("VAST_API_KEY")
    if not key:
        sys.exit("VAST_API_KEY is not set - refusing to run blind")

    started = time.time()
    previous = None
    while True:
        ts = time.time()
        gpu, net = parse(run_command(key, args.instance, COMMAND))
        total_rx = sum(v["rx_bytes"] for v in net.values())
        total_tx = sum(v["tx_bytes"] for v in net.values())
        sample = {"ts": round(ts, 1), "gpu": gpu,
                  "net_rx_bytes": total_rx, "net_tx_bytes": total_tx}
        if previous is not None:
            dt = ts - previous["ts"]
            if dt > 0:
                sample["rx_mbps"] = round((total_rx - previous["net_rx_bytes"]) * 8 / dt / 1e6, 2)
                sample["tx_mbps"] = round((total_tx - previous["net_tx_bytes"]) * 8 / dt / 1e6, 2)
        previous = sample
        with open(args.out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(sample) + "\n")
        print("[%s] gpu=%s%% mem=%s%% rx=%s Mbps tx=%s Mbps"
              % (time.strftime("%H:%M:%S"), gpu.get("gpu_util_pct"),
                 gpu.get("mem_util_pct"), sample.get("rx_mbps"), sample.get("tx_mbps")),
              flush=True)
        if args.duration and (ts - started) > args.duration * 60:
            return 0
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    sys.exit(main())
