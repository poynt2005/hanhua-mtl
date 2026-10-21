"""Translate an explicit list of archives on an already-rented machine.

  python tools/run_list.py --manifest batch-25691020.json --config config.json
  python tools/run_list.py --manifest batch-25691020.json --config config.json --destroy

Why this exists next to run_batch.py: run_batch identifies a volume by its page
count, which is ambiguous once a batch contains several books of the same length
(this batch has two 242-page books). Here every archive is named explicitly.

Guarantees:
  * never rents anything -- rent with `hanhua.py vast up` first;
  * an archive is re-run only if its output is missing or fails verification, so
    an interrupted batch resumes where it stopped;
  * on a failed archive the machine is KEPT (it is the debugging environment);
  * the machine is destroyed only with --destroy, and only after every archive
    in the list verified.

Each archive appends one line to the run log (seconds, pages, seconds/page,
output bytes), which is what the cost report is built from.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from io import BytesIO

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

PAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")


def say(msg):
    print("[run %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def touch_heartbeat(path):
    """Tell tools/heartbeat_guard.py that the batch is still making progress."""
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("%d\n" % int(time.time()))
        os.utime(path, None)
    except OSError:
        pass


def pages(path):
    with zipfile.ZipFile(path) as zf:
        return [n for n in zf.namelist()
                if not n.endswith("/") and os.path.splitext(n)[1].lower() in PAGE_EXT]


def output_of(archive, outdir=None):
    name = os.path.basename(archive)
    stem = os.path.splitext(name)[0]
    folder = outdir or os.path.dirname(archive)
    return os.path.join(folder, stem + "_MTL_zhTW.zip")


def verify(archive, outdir=None):
    """Same order, every page present, unchanged dimensions, text actually translated."""
    from PIL import Image

    out = output_of(archive, outdir)
    if not os.path.exists(out):
        say("VERIFY FAIL: no output for %s" % os.path.basename(archive))
        return False
    a, b = pages(archive), pages(out)
    if a != b:
        say("VERIFY FAIL: page list differs (in=%d out=%d)" % (len(a), len(b)))
        return False
    bad = []
    with zipfile.ZipFile(archive) as zs, zipfile.ZipFile(out) as zo:
        for name in a:
            try:
                sa = Image.open(BytesIO(zs.read(name))).size
                sb = Image.open(BytesIO(zo.read(name))).size
            except Exception as exc:  # noqa: BLE001
                bad.append((name, "unreadable: %s" % exc))
                continue
            if sa != sb:
                bad.append((name, sa, sb))
    pj = out + ".project.json"
    blocks = translated = pages_with_text = 0
    if os.path.exists(pj):
        with open(pj, encoding="utf-8") as fh:
            data = json.load(fh)
        for page in data.get("pages") or []:
            bl = page.get("blocks") or []
            if any((x.get("ocr") or "").strip() for x in bl):
                pages_with_text += 1
            blocks += len(bl)
            translated += sum(1 for x in bl if (x.get("translation") or "").strip())
    coverage = (100.0 * translated / blocks) if blocks else 100.0
    ok = not bad and (blocks == 0 or coverage >= 99.0)
    say("VERIFY %-40s in=%d out=%d order=%s dim=%s text_pages=%d blocks=%d translated=%d (%.1f%%) -> %s"
        % (os.path.basename(archive)[:40], len(a), len(b), a == b,
           "ok" if not bad else bad[:2], pages_with_text, blocks, translated,
           coverage, "OK" if ok else "PROBLEM"))
    return ok


def reset_services(cfg, ssh_host, ssh_port, ssh_key, log=print):
    """Make sure the OCR service is alive before an archive starts.

    The OCR service's VRAM grows over a session (0.4 GB at boot, 17 GB after 38
    large pages) and never gives it back, so the pipeline stops it after every
    extraction and restarts it here. That is why this both starts the restart
    loop when it is missing and kills the worker.
    """
    import urllib.request

    ocr_url = cfg.get("ocr_url") or ""
    host_port = ocr_url.split("//")[-1].split("/")[0]
    ocr_host, _, ocr_port = host_port.partition(":")
    if not ocr_host or not ocr_port:
        return False

    known_hosts = "NUL" if os.name == "nt" else "/dev/null"
    from mth import remote

    remote.write_loop(ssh_host, ssh_port, ssh_key)
    remote.stop_ocr(ssh_host, ssh_port, ssh_key)
    remote.start_ocr(ssh_host, ssh_port, ssh_key)

    health = "http://%s:%s/health" % (ocr_host, ocr_port)
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=5) as resp:
                if resp.status == 200:
                    log("reset: services back up")
                    return True
        except Exception:  # noqa: BLE001 - still loading
            pass
        time.sleep(5)
    log("reset: OCR did not come back within 180s")
    return False


def stop_ocr(ssh_host, ssh_port, ssh_key, log=print):
    """Stop the OCR service so the translation model can have the GPU.

    After extraction the service holds most of the card; loading the 14B model on
    top of that does not fit, and the run stalls on a request that can never be
    served. Called between the extract and translate stages.
    """
    known_hosts = "NUL" if os.name == "nt" else "/dev/null"
    remote = ("pkill -f 'PORT=808[0] python /root/ocr_server.py'; sleep 1; "
              "pkill -x -f 'python /root/ocr_server.py'; sleep 3; "
              "nvidia-smi --query-gpu=memory.used --format=csv,noheader")
    ssh = ["ssh", "-i", ssh_key, "-p", str(ssh_port),
           "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=" + known_hosts,
           "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "root@" + ssh_host,
           remote]
    res = subprocess.run(ssh, capture_output=True, text=True)
    log("ocr stopped after extract (vram now %s)" % (res.stdout or "").strip()[:20])
    return res.returncode == 0


def run_one(archive, config, python_exe, outdir=None, glossary=None, auto_terms=False,
            ocr="http", inpaint="lama", skip_pages="", reset=None,
            reset_target=None, ssh_key=None, heartbeat=None):
    out = output_of(archive, outdir)
    if os.path.exists(out) and verify(archive, outdir):
        say("already done: %s" % os.path.basename(out)[:60])
        return True, None
    pages_n = len(pages(archive))
    if reset:
        reset()
    say("RUN %d pages: %s" % (pages_n, os.path.basename(archive)[:60]))
    cmd = [python_exe, os.path.join(REPO, "hanhua.py"), "auto",
           "-i", archive, "-o", out,
           "--ocr", ocr, "--profile", "sakura",
           "--inpaint", inpaint, "--config", config]
    if glossary:
        cmd += ["--glossary", glossary]
    elif auto_terms:
        cmd += ["--auto-terms"]
    if skip_pages:
        cmd += ["--skip-pages", skip_pages]
    t0 = time.time()
    env = os.environ.copy()
    if reset_target:
        env["MTH_OCR_STOP_SSH"] = reset_target
        env["MTH_SSH_KEY"] = ssh_key or os.path.join(REPO, "vast_ssh")
    proc = subprocess.Popen(cmd, cwd=REPO, env=env)
    # Keep the heartbeat moving while the archive runs: an archive can take the
    # better part of an hour, and the guard must not mistake that for a dead run.
    while proc.poll() is None:
        touch_heartbeat(heartbeat)
        time.sleep(30)
    rc = proc.returncode
    secs = time.time() - t0
    if rc != 0:
        say("FAILED rc=%d after %.0fs - MACHINE KEPT for debugging" % (rc, secs))
        return False, {"archive": archive, "pages": pages_n, "seconds": round(secs, 1),
                       "rc": rc, "verified": False}
    record = {"archive": os.path.basename(archive), "pages": pages_n,
              "seconds": round(secs, 1),
              "sec_per_page": round(secs / max(1, pages_n), 2),
              "output_bytes": os.path.getsize(out) if os.path.exists(out) else 0,
              "verified": verify(archive, outdir)}
    say("%d pages in %.0fs (%.1f s/page) -> %s"
        % (pages_n, secs, secs / max(1, pages_n), "verified" if record["verified"] else "NOT VERIFIED"))
    return record["verified"], record


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="JSON list of archives (see tools/README.md)")
    ap.add_argument("--config", default=os.path.join(REPO, "config.json"))
    ap.add_argument("--log", default="run_list.jsonl", help="append per-archive timings here")
    ap.add_argument("--outdir", default="",
                    help="write outputs here instead of next to the input (saves disk)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--ocr", default="http", help="ocr backend passed to auto (http on the GPU box)")
    ap.add_argument("--inpaint", default="lama", help="inpaint backend passed to auto")
    ap.add_argument("--reset-gpu", action="store_true",
                    help="restart the OCR service and the model between archives "
                         "(its VRAM never shrinks on its own)")
    ap.add_argument("--ssh-host", help="ssh host for --reset-gpu (the instance, not the proxy)")
    ap.add_argument("--ssh-port", type=int, help="ssh port for --reset-gpu")
    ap.add_argument("--ssh-key", default=os.path.join(REPO, "vast_ssh"),
                    help="ssh key for --reset-gpu")
    ap.add_argument("--heartbeat", default=os.path.join(REPO, "batch_heartbeat.txt"),
                    help="file touched after every archive; tools/heartbeat_guard.py "
                         "destroys the machine when it stops moving")
    ap.add_argument("--limit", type=int, default=0, help="stop after N archives (for a pilot)")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed archive")
    ap.add_argument("--destroy", action="store_true",
                    help="destroy the recorded instance after all archives verified")
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as fh:
        manifest = json.load(fh)
    raw = manifest["archives"] if isinstance(manifest, dict) else manifest
    archives = [item if isinstance(item, dict) else {"archive": item} for item in raw]
    if args.limit:
        archives = archives[:args.limit]

    say("%d archive(s), config=%s" % (len(archives), os.path.basename(args.config)))

    outdir = os.path.abspath(args.outdir) if args.outdir else None
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    reset = None
    reset_target = None
    if args.reset_gpu:
        if not args.ssh_host or not args.ssh_port:
            say("--reset-gpu needs --ssh-host and --ssh-port; continuing without reset")
        else:
            with open(args.config, encoding="utf-8") as fh:
                base_cfg = json.load(fh)
            reset = (lambda: reset_services(base_cfg, args.ssh_host, args.ssh_port,
                                            args.ssh_key, log=say))
            reset_target = "%s:%s" % (args.ssh_host, args.ssh_port)

    done = 0
    touch_heartbeat(args.heartbeat)
    for entry in archives:
        ok, record = run_one(entry["archive"], args.config, args.python, outdir,
                             entry.get("glossary"), bool(entry.get("auto_terms")),
                             args.ocr, args.inpaint, entry.get("skip_pages", ""),
                             reset, reset_target, args.ssh_key, args.heartbeat)
        touch_heartbeat(args.heartbeat)
        if record:
            with open(args.log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        if not ok:
            if not args.keep_going:
                say("stopping at %s (use --keep-going to skip failures)"
                    % os.path.basename(entry["archive"])[:50])
                return 4
        else:
            done += 1

    say("%d/%d archives verified" % (done, len(archives)))
    if done != len(archives):
        say("not everything verified - machine left running")
        return 4

    try:
        os.remove(args.heartbeat)
    except OSError:
        pass

    if not args.destroy:
        say("--destroy not given: machine left running on purpose")
        return 0

    key = os.environ.get("VAST_API_KEY")
    if not key:
        say("VAST_API_KEY not set; cannot destroy. Destroy it yourself.")
        return 5
    from mth import vast as V
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
