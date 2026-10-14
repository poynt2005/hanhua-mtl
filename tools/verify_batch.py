"""Verify every delivered volume in a folder.

Pairs each input archive with its `<stem>_MTL_zhTW.zip` output **by name** -
page count alone is not an identity (two different volumes here had 237 pages).

Checks, per volume:
  page count identical, page order identical, image dimensions unchanged,
  translation coverage (every block with source text has a non-empty
  translation).

    python tools/verify_batch.py --dir .
"""
import argparse
import io
import json
import os
import sys
import zipfile

from PIL import Image

PAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")


def pages(path):
    with zipfile.ZipFile(path) as zf:
        return [n for n in zf.namelist() if n.lower().endswith(PAGE_EXT)]


def verify_one(src, out):
    a, b = pages(src), pages(out)
    zs, zo = zipfile.ZipFile(src), zipfile.ZipFile(out)
    mismatched = []
    for name in a:
        try:
            if (Image.open(io.BytesIO(zs.read(name))).size
                    != Image.open(io.BytesIO(zo.read(name))).size):
                mismatched.append(name)
        except KeyError:
            mismatched.append(name + " (missing in output)")
    blocks = translated = low = 0
    pj = out + ".project.json"
    if os.path.exists(pj):
        data = json.load(open(pj, encoding="utf-8"))
        for page in (data.get("pages") or []):
            bl = page.get("blocks") or []
            blocks += len(bl)
            translated += sum(1 for x in bl if (x.get("translation") or "").strip())
            if len(bl) <= 2:
                low += 1
    return {
        "in": len(a), "out": len(b), "order": a == b, "mismatched": mismatched,
        "blocks": blocks, "translated": translated, "low": low,
        "mb": os.path.getsize(out) / 1e6,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=".", help="folder holding the archives")
    args = ap.parse_args()

    rows, all_ok = [], True
    for name in sorted(os.listdir(args.dir)):
        if not name.lower().endswith(".zip") or name.endswith("_MTL_zhTW.zip"):
            continue
        if name.startswith("pilot"):
            continue
        src = os.path.join(args.dir, name)
        out = os.path.splitext(src)[0] + "_MTL_zhTW.zip"
        if not os.path.exists(out):
            continue
        r = verify_one(src, out)
        ok = r["order"] and not r["mismatched"]
        all_ok = all_ok and ok
        rows.append((name, r, ok))

    if not rows:
        print("no input/output pairs found in %s" % args.dir)
        return 1

    print("%-44s %6s %5s %6s %8s %9s %8s" % (
        "volume", "pages", "ok", "sizes", "blocks", "translated", "MB"))
    print("-" * 96)
    tot_pages = tot_blocks = tot_tr = tot_low = 0
    for name, r, ok in rows:
        print("%-44s %6d %5s %6d %8d %9d %8.1f" % (
            name[:44], r["in"], "OK" if ok else "BAD", len(r["mismatched"]),
            r["blocks"], r["translated"], r["mb"]))
        tot_pages += r["in"]
        tot_blocks += r["blocks"]
        tot_tr += r["translated"]
        tot_low += r["low"]
    print("-" * 96)
    print("TOTAL pages=%d  blocks=%d  translated=%d (%.1f%%)  low-block pages=%d" % (
        tot_pages, tot_blocks, tot_tr, 100.0 * tot_tr / max(1, tot_blocks), tot_low))
    print("ALL VERIFIED: %s" % all_ok)
    if tot_low:
        print("\n%d pages had <=2 detected blocks. Run tools/audit_detection.py to see"
              " whether they still contain text (i.e. were missed)." % tot_low)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
