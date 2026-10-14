"""Rent ONE machine, PROVE its pull is fast enough, then gate + run + verify.

This is the whole flow in one command, and it is the recommended entry point:

    python tools/rent_and_run.py --yes --image poynt2005/hanhua-gpu:0.5.0 \
           --tag 0.5.0 --config config.json --dir . --run 250 237

Why it exists (all measured, see llm_handoff/15_run_log.md):

  * the advertised bandwidth filter is not evidence - a 381 Mbps host pulled at
    1.1 MB/s. So every candidate is measured with tools/pull_watch.py, which
    destroys + blacklists a bad host and moves on;
  * a 12 GB layer downloads for ~10 minutes with NO status change, so "slow" and
    "stuck" are indistinguishable from the status message alone - only a measured
    rate decides;
  * `hanhua.py vast up` alone would have destroyed a good machine mid-pull
    (its 25-minute cap), which is why the rent loop lives here;
  * one machine at a time: it refuses to rent while one is already recorded.

It never leaves a machine behind: a bad host is destroyed by pull_watch, a gate
failure KEEPS the machine for debugging (and says so), and a verified run tears
it down.
"""
import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from mth import vast as V  # noqa: E402

ONSTART = ("nohup bash /root/container_start.sh >/root/start.log 2>&1 </dev/null "
           "& echo started")


def say(msg):
    print("[rent %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="required to actually rent")
    ap.add_argument("--image", required=True, help="registry image, e.g. repo/name:tag")
    ap.add_argument("--tag", required=True, help="tag, for the pull measurement")
    ap.add_argument("--config", default=os.path.join(REPO, "config.json"))
    ap.add_argument("--dir", default=os.path.dirname(REPO),
                    help="folder holding the input archives")
    ap.add_argument("--run", type=int, action="append", default=[],
                    help="page count of a volume to translate (repeatable)")
    ap.add_argument("--wait", type=int, action="append", default=[],
                    help="page count of a volume being processed elsewhere")
    ap.add_argument("--text-model", default="sakura-qwen3-q6k")
    ap.add_argument("--max-price", type=float, default=0.28)
    ap.add_argument("--min-vram", type=int, default=24)
    ap.add_argument("--min-down", type=int, default=200)
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--per-attempt", type=int, default=8)
    ap.add_argument("--disk", type=int, default=60)
    ap.add_argument("--no-destroy", action="store_true")
    args = ap.parse_args()

    if not args.yes:
        raise SystemExit("refusing to rent without --yes (see deploy/VASTAI_PLAYBOOK.md R1)")
    key = os.environ.get("VAST_API_KEY")
    if not key:
        raise SystemExit("VAST_API_KEY is not set")

    if V.get_current(REPO):
        raise SystemExit("an instance is already recorded as current "
                         "(vast_current.txt); refusing to rent another")

    for attempt in range(args.attempts):
        offers = V.search_offers(key, min_vram=args.min_vram, min_down=args.min_down,
                                 min_up=50, max_price=args.max_price, order="cheap",
                                 limit=40)
        # advertised bandwidth is a weak but real prior (381 Mbps -> 1.1 MB/s,
        # 812 -> 21 MB/s, 1000+ -> 50 MB/s); the measured rate still decides.
        offers.sort(key=lambda o: -float(o.get("inet_down") or 0))
        blacklist = V.load_blacklist(REPO)
        say("attempt %d: %d offers (best advertised bandwidth first)" % (attempt + 1, len(offers)))

        for offer in offers[:args.per_attempt]:
            host = offer.get("public_ipaddr")
            if host in blacklist:
                continue
            say("  try %s %s @ $%.3f host=%s" % (offer.get("id"), offer.get("gpu_name"),
                                                 offer.get("dph_total", 0), host))
            iid = V.create_instance(key, offer["id"], image=args.image, disk=args.disk,
                                    onstart_cmd=ONSTART)
            if not iid:
                continue
            V.set_current(REPO, iid)
            say("  instance %s created; measuring the pull" % iid)

            rc = subprocess.run([sys.executable, os.path.join(REPO, "tools", "pull_watch.py"),
                                 "--instance", str(iid), "--tag", args.tag,
                                 "--registry", args.image.split(":")[0],
                                 "--destroy"], cwd=REPO).returncode
            if rc != 0:
                say("  pull verdict BAD (rc=%d) - host blacklisted, next offer" % rc)
                continue

            info = V.instance_info(key, iid)
            host = info.get("public_ipaddr")
            ports = info.get("ports", {})
            sakura = ports.get("8000/tcp", [{}])[0].get("HostPort")
            ocr = ports.get("8080/tcp", [{}])[0].get("HostPort")
            say("  pull OK; waiting for endpoints (sakura=%s ocr=%s)" % (sakura, ocr))
            if not sakura or not ocr or not V.wait_http(host, sakura, ocr, timeout=1800, log=say):
                say("  endpoints never came up - destroying, next offer")
                V.destroy(key, iid)
                V.add_blacklist(REPO, host)
                V.clear_current(REPO)
                blacklist = V.load_blacklist(REPO)
                continue

            cfg = {}
            if os.path.exists(args.config):
                with open(args.config, encoding="utf-8") as fh:
                    cfg = json.load(fh)
            cfg.update({"instance_id": iid,
                        "api_base": "http://%s:%s/v1" % (host, sakura),
                        "ocr_url": "http://%s:%s/prepare" % (host, ocr),
                        "text_model": args.text_model})
            with open(args.config, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh, ensure_ascii=False, indent=2)
            say("  wrote %s" % args.config)

            say("  GATE")
            if subprocess.run([sys.executable, os.path.join(REPO, "tools", "gate.py"),
                               "--config", args.config], cwd=REPO).returncode != 0:
                say("  GATE FAILED - MACHINE KEPT for debugging (instance %s)" % iid)
                return 6

            cmd = [sys.executable, os.path.join(REPO, "tools", "run_batch.py"),
                   "--config", args.config, "--dir", os.path.abspath(args.dir)]
            for count in args.wait:
                cmd += ["--wait", str(count)]
            for count in args.run:
                cmd += ["--run", str(count)]
            if args.no_destroy:
                cmd += ["--no-destroy"]
            return subprocess.run(cmd, cwd=REPO).returncode

    say("no host passed the pull check")
    return 3


if __name__ == "__main__":
    sys.exit(main())
