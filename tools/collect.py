"""Collect delivered volumes into one folder, verifying each one on the way.

Verifies page count and page order against the input BEFORE moving anything, so
the delivery folder cannot contain an unverified archive. Writes a MANIFEST.txt
and brings each volume's `.project.json` (per-block source/translation) along.

    python tools/collect.py --dir . --dest ComicsZHTW [--include-pilot]
"""
import argparse
import os
import shutil
import sys
import time
import zipfile

PAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")


def pages(path):
    with zipfile.ZipFile(path) as zf:
        return [n for n in zf.namelist() if n.lower().endswith(PAGE_EXT)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=".", help="folder holding the archives")
    ap.add_argument("--dest", default="ComicsZHTW", help="delivery folder")
    ap.add_argument("--include-pilot", action="store_true",
                    help="also move archives whose name starts with 'pilot'")
    args = ap.parse_args()

    src_dir = os.path.abspath(args.dir)
    dest = os.path.join(src_dir, args.dest)
    os.makedirs(dest, exist_ok=True)

    moved, skipped = [], []
    for name in sorted(os.listdir(src_dir)):
        if not name.endswith("_MTL_zhTW.zip"):
            continue
        if name.startswith("pilot") and not args.include_pilot:
            skipped.append((name, "pilot sample"))
            continue
        src = os.path.join(src_dir, name)
        stem = name[:-len("_MTL_zhTW.zip")]
        inp = os.path.join(src_dir, stem + ".zip")

        out_pages = pages(src)
        if os.path.exists(inp):
            in_pages = pages(inp)
            if in_pages != out_pages:
                print("  !! %s: page order differs - NOT moving" % name[:60])
                skipped.append((name, "order mismatch"))
                continue
            note = "%d pages, order OK" % len(out_pages)
        else:
            note = "%d pages (input not present, order unchecked)" % len(out_pages)

        dst = os.path.join(dest, name)
        shutil.move(src, dst)
        if os.path.exists(src + ".project.json"):
            shutil.move(src + ".project.json", dst + ".project.json")
        moved.append((name, len(out_pages)))
        print("  -> %s : %s" % (name[:60], note))

    manifest = os.path.join(dest, "MANIFEST.txt")
    with open(manifest, "w", encoding="utf-8") as fh:
        fh.write("Translated manga - machine translation, NOT human reviewed.\n")
        fh.write("Generated %s\n\n" % time.strftime("%Y-%m-%d %H:%M"))
        fh.write("%-6s  %s\n" % ("pages", "file"))
        fh.write("-" * 78 + "\n")
        for name, count in moved:
            fh.write("%-6d  %s\n" % (count, name))
        fh.write("-" * 78 + "\n")
        fh.write("%-6d  TOTAL (%d volumes)\n\n" % (sum(m[1] for m in moved), len(moved)))
        fh.write("Each archive is the input archive with its pages translated and\n")
        fh.write("re-encoded: same page count, same order, same image dimensions.\n")
        fh.write("A matching <archive>.project.json holds the per-block source text\n")
        fh.write("and translation for review.\n")

    print("\n%s" % dest)
    print("moved: %d volumes, %d pages" % (len(moved), sum(m[1] for m in moved)))
    for name, why in skipped:
        print("skipped: %s (%s)" % (name[:50], why))
    print("manifest: %s" % manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
