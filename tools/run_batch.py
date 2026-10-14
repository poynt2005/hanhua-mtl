"""Run volumes sequentially on an already-rented machine, verify each, tear down.

  python tools/run_batch.py --config config.json --wait 257 --run 16 250

  --wait N   a volume someone else is already processing: wait for its output
  --run  N   volumes to translate here, in order
  N is the page count, used to find the input archive. Inputs that already have
  an output are skipped when running (page counts collide), but NOT when waiting
  (the volume being waited on legitimately has one by the time we look).

Behaviour that matters:
  * the machine is destroyed **only after every volume verified**;
  * on any failure the machine is KEPT (it is the debugging environment);
  * it never rents anything - rent with `hanhua.py vast up` first.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from mth import vast as V  # noqa: E402

PAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")


def say(msg):
    print("[batch %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def pages(path):
    with zipfile.ZipFile(path) as zf:
        return [n for n in zf.namelist() if n.lower().endswith(PAGE_EXT)]


def output_of(archive):
    return os.path.splitext(archive)[0] + "_MTL_zhTW.zip"


def find_input(folder, count, skip_done=True):
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(".zip") or name.endswith("_MTL_zhTW.zip"):
            continue
        if name.startswith("pilot"):
            continue
        path = os.path.join(folder, name)
        if skip_done and os.path.exists(output_of(path)):
            continue
        try:
            if len(pages(path)) == count:
                return path
        except Exception:  # noqa: BLE001
            continue
    return None


def verify(archive):
    out = output_of(archive)
    if not os.path.exists(out):
        say("VERIFY FAIL: no output for %s" % os.path.basename(archive))
        return False
    from io import BytesIO
    from PIL import Image
    a, b = pages(archive), pages(out)
    zs, zo = zipfile.ZipFile(archive), zipfile.ZipFile(out)
    bad = [n for n in a
           if Image.open(BytesIO(zs.read(n))).size != Image.open(BytesIO(zo.read(n))).size]
    pj = out + ".project.json"
    blocks = translated = 0
    if os.path.exists(pj):
        data = json.load(open(pj, encoding="utf-8"))
        bl = [x for p in (data.get("pages") or []) for x in (p.get("blocks") or [])]
        blocks = len(bl)
        translated = sum(1 for x in bl if (x.get("translation") or "").strip())
    ok = (a == b) and not bad
    say("VERIFY %s: in=%d out=%d same_order=%s size_mismatch=%s blocks=%d translated=%d -> %s"
        % (os.path.basename(archive)[:34], len(a), len(b), a == b, bad or "none",
           blocks, translated, "OK" if ok else "PROBLEM"))
    return ok


def wait_output(folder, count, timeout):
    src = find_input(folder, count, skip_done=False)
    if not src:
        say("no input with %d pages found" % count)
        return False
    out = output_of(src)
    say("waiting for %s" % os.path.basename(out)[:50])
    start, last_size, stable = time.time(), -1, 0
    while time.time() - start < timeout:
        if os.path.exists(out):
            size = os.path.getsize(out)
            try:
                n = len(pages(out))
            except Exception:  # noqa: BLE001
                n = -1
            if n == count and size == last_size:
                stable += 1
                if stable >= 3:
                    say("archive complete: %d pages, %.1f MB" % (n, size / 1e6))
                    return True
            elif size != last_size:
                stable, last_size = 0, size
                say("writing archive: %d/%d pages, %.1f MB" % (n, count, size / 1e6))
        time.sleep(20)
    say("timed out waiting for %d pages" % count)
    return False


def run_volume(folder, config, count, python_exe):
    src = find_input(folder, count)
    if not src:
        say("no input with %d pages found (already done?)" % count)
        return False
    out = output_of(src)
    if os.path.exists(out):
        say("output already present for %d pages; verifying it instead" % count)
        return verify(src)
    say("RUN %d pages: %s" % (count, os.path.basename(src)[:44]))
    t0 = time.time()
    rc = subprocess.run([python_exe, os.path.join(REPO, "hanhua.py"), "auto",
                         "-i", src, "--ocr", "http", "--profile", "sakura",
                         "--config", config], cwd=REPO).returncode
    say("finished rc=%d in %.0fs (%.1f s/page)" % (rc, time.time() - t0,
                                                   (time.time() - t0) / max(1, count)))
    return rc == 0 and verify(src)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=os.path.join(REPO, "config.json"))
    ap.add_argument("--dir", default=os.path.dirname(REPO), help="folder with the archives")
    ap.add_argument("--wait", type=int, action="append", default=[],
                    help="page count of a volume being processed elsewhere")
    ap.add_argument("--run", type=int, action="append", default=[],
                    help="page count of a volume to translate here")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--wait-timeout", type=int, default=7200)
    ap.add_argument("--no-destroy", action="store_true",
                    help="leave the machine running at the end")
    args = ap.parse_args()

    key = os.environ.get("VAST_API_KEY")
    folder = os.path.abspath(args.dir)

    for count in args.wait:
        if not wait_output(folder, count, args.wait_timeout):
            say("wait volume failed - machine kept")
            return 3
        if not verify(find_input(folder, count, skip_done=False)):
            say("wait volume failed verification - machine kept")
            return 3

    for count in args.run:
        if not run_volume(folder, args.config, count, args.python):
            say("volume %d failed - MACHINE KEPT for debugging" % count)
            return 4

    say("ALL VOLUMES VERIFIED")
    if args.no_destroy:
        say("--no-destroy given; machine left running")
        return 0
    if not key:
        say("VAST_API_KEY not set; cannot destroy. Destroy it yourself.")
        return 5
    iid = V.get_current(REPO)
    say("destroying instance %s" % iid)
    if iid:
        V.destroy(key, iid)
    V.clear_current(REPO)
    time.sleep(6)
    say("remaining instances: %s" % [i.get("id") for i in V.list_instances(key)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
