"""Keep at most ONE instance of ours alive.

Run this in the background whenever a run is in flight. It destroys any instance
carrying our label that is NOT the recorded current one, so a leftover script or
a half-finished teardown can never leave two machines billing. It never touches
instances that do not carry our label (the user's own machines are safe).

Exit codes: 0 only if killed; it loops forever otherwise.
"""
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from mth import vast as V  # noqa: E402

INTERVAL = int(os.environ.get("WATCH_INTERVAL", "60"))


def main():
    key = os.environ.get("VAST_API_KEY")
    if not key:
        print("VAST_API_KEY not set", flush=True)
        return 2

    def log(msg):
        print("[watch %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)

    log("started; interval=%ds" % INTERVAL)
    while True:
        try:
            current = V.get_current(REPO)
            mine = V.owned_instances(key)
            ids = [i.get("id") for i in mine]
            if current is None:
                # No recorded owner: never destroy anything automatically, just
                # shout (destroying here could kill a legitimate run).
                if len(mine) > 1:
                    log("!! %d instances of ours and no recorded owner: %s "
                        "- decide manually" % (len(mine), ids))
                elif mine:
                    log("note: %s is running but not recorded as current" % ids)
            else:
                if len(mine) > 1:
                    log("!! %d instances of ours: %s (allowed: %s)" % (len(mine), ids, current))
                removed = V.enforce_single(key, REPO, keep=current, log=log)
                if removed:
                    log("removed extras: %s" % removed)
        except Exception as exc:  # noqa: BLE001
            log("check failed: %s" % exc)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    sys.exit(main())
