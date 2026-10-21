"""Destroy the rented machine when the batch stops making progress.

Why this exists: on 2026-10-21 the local runner process was killed at 01:41 while
an RTX 3090 kept billing until 07:49 (~$1.29 of nothing). The runner was also the
only thing that destroyed the machine, and the only thing that reported progress,
so killing it removed supervision and teardown at the same time.

This guard is meant to be run by a scheduler (every 5 minutes), independent of any
chat session. It acts only when a heartbeat file exists and has stopped moving:

  * heartbeat exists and is older than --max-idle  -> destroy the instance
  * heartbeat missing                              -> do nothing (no batch running)
  * instance not recorded                          -> do nothing

  python tools/heartbeat_guard.py --repo <repo> --heartbeat <file> [--dry-run]

VAST_API_KEY must be in the environment (a user-level variable is enough).
"""
import argparse
import json
import os
import subprocess
import sys
import time

REPO_DEFAULT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_instance(repo):
    path = os.path.join(repo, "vast_current.txt")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read().strip()
    except OSError:
        return None
    if not text:
        return None
    return text.split()[0]


def log(line):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print("[%s] %s" % (stamp, line), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--heartbeat", default=os.path.join(REPO_DEFAULT, "batch_heartbeat.txt"))
    ap.add_argument("--max-idle", type=float, default=15.0,
                    help="minutes without a heartbeat before the machine is destroyed")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.heartbeat):
        return 0  # no batch is running; a manually rented machine is not our business

    age_min = (time.time() - os.path.getmtime(args.heartbeat)) / 60.0
    iid = read_instance(args.repo)
    if not iid:
        log("heartbeat exists but no instance is recorded (age %.1f min)" % age_min)
        return 0
    if age_min < args.max_idle:
        log("ok: instance %s, heartbeat %.1f min old" % (iid, age_min))
        return 0

    log("STALE: instance %s, no progress for %.1f min (limit %.0f)"
        % (iid, age_min, args.max_idle))
    if args.dry_run:
        log("dry-run: would destroy %s" % iid)
        return 0

    key = os.environ.get("VAST_API_KEY")
    if not key:
        log("VAST_API_KEY is not set: cannot destroy. Set it as a user variable.")
        return 2
    result = subprocess.run(
        [sys.executable, os.path.join(args.repo, "hanhua.py"), "vast", "down"],
        cwd=args.repo, capture_output=True, text=True,
        env=dict(os.environ, VAST_API_KEY=key))
    log("destroy rc=%d out=%s" % (result.returncode,
                                  (result.stdout or result.stderr).strip()[:120]))
    return 0 if result.returncode == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
