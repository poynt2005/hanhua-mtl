"""Read-only vast.ai account check. Prints credit / total_spend / instances.

Usage:  set VAST_API_KEY, then  python deploy/vast_account.py
Key is read from the environment only and never written to disk.
"""
import json
import os
import sys
import urllib.error
import urllib.request

BASE = "https://console.vast.ai"


def call(path, key):
    req = urllib.request.Request(BASE + path)
    req.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    key = os.environ.get("VAST_API_KEY", "")
    if not key:
        sys.exit("set VAST_API_KEY in the environment")
    print("key length: %d (vast keys are 64 chars)" % len(key))

    user = None
    last_err = None
    for path in ("/api/v0/users/current/", "/api/v0/users/current"):
        try:
            user = call(path, key)
            break
        except urllib.error.HTTPError as err:
            last_err = "HTTP %s: %s" % (err.code, err.read().decode("utf-8", "replace")[:200])
        except urllib.error.URLError as err:
            last_err = "network error: %s" % err
    if user is None:
        sys.exit("auth failed -> %s" % last_err)

    print("id:          %s" % user.get("id"))
    print("email:       %s" % user.get("email"))
    print("credit:      %s" % user.get("credit"))
    print("total_spend: %s" % user.get("total_spend"))

    try:
        inst = call("/api/v0/instances/", key)
        items = inst.get("instances", inst) if isinstance(inst, dict) else inst
        print("instances:   %d" % len(items))
        for it in items:
            print("   - id=%s gpu=%s status=%s dph=%s" % (
                it.get("id"), it.get("gpu_name"), it.get("actual_status"), it.get("dph_total")))
    except Exception as err:  # noqa: BLE001
        print("instances:   (could not list: %s)" % err)


if __name__ == "__main__":
    main()
