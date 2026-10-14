"""Time the OCR service's /prepare on real pages from an archive.

Useful for deciding whether a page can exceed a host's idle timeout, and for
sizing a batch. Read-only: it posts a handful of pages and prints the timings.

    python tools/prep_time.py --config config.json --dir . --pages 250 --count 4
"""
import argparse
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile

from PIL import Image

PAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")


def pages(path):
    with zipfile.ZipFile(path) as zf:
        return [n for n in zf.namelist() if n.lower().endswith(PAGE_EXT)]


def find_input(folder, count):
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(".zip") or name.endswith("_MTL_zhTW.zip") or name.startswith("pilot"):
            continue
        path = os.path.join(folder, name)
        try:
            if len(pages(path)) == count:
                return path
        except Exception:  # noqa: BLE001
            continue
    return None


def post(url, raw, timeout=300):
    boundary = "----preptime"
    body = b""
    body += ("--%s\r\n" % boundary).encode()
    body += b'Content-Disposition: form-data; name="file"; filename="p.jpg"\r\n'
    body += b"Content-Type: application/octet-stream\r\n\r\n" + raw + b"\r\n"
    body += ("--%s--\r\n" % boundary).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "multipart/form-data; boundary=%s" % boundary)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return time.time() - t0, len(data.get("blocks") or []), None
    except urllib.error.HTTPError as exc:
        return time.time() - t0, 0, "HTTP %s" % exc.code
    except Exception as exc:  # noqa: BLE001
        return time.time() - t0, 0, type(exc).__name__


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--dir", default=".")
    ap.add_argument("--pages", type=int, default=250, help="page count of the archive to sample")
    ap.add_argument("--count", type=int, default=4, help="how many pages to time")
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8")) if os.path.exists(args.config) else {}
    url = cfg.get("ocr_prepare_url") or cfg.get("ocr_url") or ""
    if not url:
        print("no ocr_url in %s" % args.config)
        return 2
    print("ocr: %s" % url)

    src = find_input(os.path.abspath(args.dir), args.pages)
    if not src:
        print("no archive with %d pages in %s" % (args.pages, args.dir))
        return 2

    with zipfile.ZipFile(src) as zf:
        names = pages(zf)
        step = max(1, len(names) // max(1, args.count))
        picks = names[::step][:args.count]
        print("%s: %d pages; timing %d of them" % (os.path.basename(src)[:50], len(names), len(picks)))
        times = []
        for name in picks:
            raw = zf.read(name)
            with Image.open(io.BytesIO(raw)) as im:
                size = im.size
            secs, blocks, err = post(url, raw)
            times.append(secs)
            print("  %-14s %5dx%-5d %6.0f KB  %5.1fs  blocks=%d %s"
                  % (name, size[0], size[1], len(raw) / 1024, secs, blocks, err or ""))
    if times:
        print("\nmean %.1fs/page, max %.1fs/page" % (sum(times) / len(times), max(times)))
        print("(if the max approaches a host's idle timeout, /prepare can be dropped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
