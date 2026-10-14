"""Watch an image pull and decide, early, whether the host is worth keeping.

Why this exists: the advertised bandwidth filter is not evidence. A host was
kept for 35 minutes ($0.125) before its pull was declared bad - and the existing
guard could not catch it, because a 12 GB layer downloads for ~10 minutes with
NO status change (so "slow" looks exactly like "stuck").

How it measures (no SSH needed - SSH is closed during the pull):
  * fetch the image manifest from the registry -> layer digest -> size
  * poll the instance's status_msg and count which layers have completed
  * report: layers done / total, bytes done, effective MB/s, and how long since
    the last layer completed

Verdict:
  * `running`                                    -> exit 0
  * "Retrying" persists longer than --retry-grace -> exit 2 (flaky registry route)
  * no layer completes for longer than --max-silence while unfinished
                                                 -> exit 3 (effectively stalled)

    python tools/pull_watch.py --instance 12345678 [--destroy] [--registry poynt2005/hanhua-gpu] --tag 0.5.0
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from mth import vast as V  # noqa: E402


def say(msg):
    print("[pull %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def fetch_manifest(registry, tag):
    """Return {short_digest: size} for the linux/amd64 image."""
    def get(url, headers=None, token=None):
        req = urllib.request.Request(url)
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        if token:
            req.add_header("Authorization", "Bearer " + token)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    accept = ("application/vnd.docker.distribution.manifest.v2+json,"
              "application/vnd.oci.image.manifest.v1+json,"
              "application/vnd.docker.distribution.manifest.list.v2+json,"
              "application/vnd.oci.image.index.v1+json")
    token = get("https://auth.docker.io/token?service=registry.docker.io"
                "&scope=repository:%s:pull" % registry)["token"]
    man = get("https://registry-1.docker.io/v2/%s/manifests/%s" % (registry, tag),
              {"Accept": accept}, token)
    if "manifests" in man:
        pick = next((m for m in man["manifests"]
                     if (m.get("platform") or {}).get("architecture") == "amd64"), None)
        if pick:
            man = get("https://registry-1.docker.io/v2/%s/manifests/%s"
                      % (registry, pick["digest"]), {"Accept": accept}, token)
    return {l["digest"].split(":")[1][:12]: l["size"] for l in man.get("layers", [])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instance", type=int, required=True)
    ap.add_argument("--registry", default="poynt2005/hanhua-gpu")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--interval", type=int, default=20)
    ap.add_argument("--retry-grace", type=int, default=180,
                    help="seconds of continuous 'Retrying' before giving up")
    ap.add_argument("--max-silence", type=int, default=780,
                    help="seconds with no layer completing before giving up")
    ap.add_argument("--destroy", action="store_true",
                    help="destroy the instance when the verdict is bad")
    args = ap.parse_args()

    key = os.environ.get("VAST_API_KEY")
    layers = fetch_manifest(args.registry, args.tag)
    total = sum(layers.values())
    say("image %s:%s -> %d layers, %.2f GB" % (args.registry, args.tag, len(layers), total / 1e9))
    say("biggest layers: %s" % ", ".join(
        "%s=%.2fGB" % (k, v / 1e9) for k, v in sorted(layers.items(), key=lambda kv: -kv[1])[:3]))

    seen, done_bytes = set(), 0
    start = time.time()
    last_progress = start
    retry_since = None
    last_msg = None

    while True:
        info = V.instance_info(key, args.instance)
        status = info.get("actual_status")
        msg = (info.get("status_msg") or "").strip()
        now = time.time()

        if status == "running":
            say("PULL FINISHED after %.1f min" % ((now - start) / 60))
            return 0
        if not info:
            say("instance is gone")
            return 4

        for short in re.findall(r"([0-9a-f]{12}): (?:Pull complete|Download complete)", msg):
            if short in layers and short not in seen:
                seen.add(short)
                done_bytes += layers[short]
                last_progress = now

        if msg != last_msg:
            el = now - start
            rate = (done_bytes / 1e6) / max(1.0, el)
            pct = 100.0 * done_bytes / total if total else 0
            say("%5.1f min | %d/%d layers | %5.2f/%.2f GB (%.0f%%) | %.1f MB/s avg | %s"
                % (el / 60, len(seen), len(layers), done_bytes / 1e9, total / 1e9, pct, rate,
                   msg[:60].replace("\n", " / ")))
            last_msg = msg

        if "retrying" in msg.lower():
            retry_since = retry_since or now
            if now - retry_since > args.retry_grace:
                say("VERDICT: BAD - still retrying after %ds (flaky registry route)" % args.retry_grace)
                break
        else:
            retry_since = None

        if now - last_progress > args.max_silence and len(seen) < len(layers):
            say("VERDICT: BAD - no layer completed for %.0f min (effectively stalled)"
                % ((now - last_progress) / 60))
            break

        time.sleep(args.interval)

    if args.destroy and key:
        host = V.instance_info(key, args.instance).get("public_ipaddr")
        say("destroying %s (host %s)" % (args.instance, host))
        V.destroy(key, args.instance)
        if host:
            V.add_blacklist(REPO, host)
        V.clear_current(REPO)
    else:
        say("not destroying (--destroy not given); destroy it yourself")
    return 2


if __name__ == "__main__":
    sys.exit(main())
