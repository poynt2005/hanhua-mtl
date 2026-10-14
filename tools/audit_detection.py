"""Audit detection quality: did any delivered page keep its source text?

Method (free - numpy + opencv only, no GPU, no viewing, no money):
  for every page, measure "text-likeness" = Otsu dark-pixel ratio + the number of
  connected components sized like text strokes (area 3-2000, w/h <= 120), for
  BOTH the input page and the delivered page.

  If the delivered page still has the same text-likeness as the source, then
  nothing was erased or translated there - the detector missed that page.

This is the check that found 74 of 1034 delivered pages still carrying text
(one volume had 20 consecutive missed pages). It is also the before/after
measurement to use when changing detection.

    python tools/audit_detection.py --dir .
"""
import argparse
import io
import json
import os
import sys
import zipfile

import cv2
import numpy as np

PAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")


def strokes(data):
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        return None
    _, bw = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    count, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    small = 0
    for i in range(1, count):
        _x, _y, w, h, area = stats[i]
        if 3 <= area <= 2000 and 2 <= w <= 120 and 2 <= h <= 120:
            small += 1
    return small


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=".", help="folder holding the archives")
    ap.add_argument("--min-kept", type=float, default=0.7,
                    help="output/input stroke ratio above which the page looks unchanged")
    ap.add_argument("--min-strokes", type=int, default=300,
                    help="minimum output strokes for the page to count as 'has text'")
    args = ap.parse_args()

    total_low = total_still = 0
    for name in sorted(os.listdir(args.dir)):
        if not name.lower().endswith(".zip") or name.endswith("_MTL_zhTW.zip") or name.startswith("pilot"):
            continue
        src = os.path.join(args.dir, name)
        out = os.path.splitext(src)[0] + "_MTL_zhTW.zip"
        pj = out + ".project.json"
        if not (os.path.exists(out) and os.path.exists(pj)):
            continue

        data = json.load(open(pj, encoding="utf-8"))
        counts = [len(p.get("blocks") or []) for p in (data.get("pages") or [])]
        zs, zo = zipfile.ZipFile(src), zipfile.ZipFile(out)
        names_s = [n for n in zs.namelist() if n.lower().endswith(PAGE_EXT)]
        names_o = [n for n in zo.namelist() if n.lower().endswith(PAGE_EXT)]

        print("\n%s" % name[:70])
        print("  pages=%d  avg blocks/page=%.1f" % (len(counts), sum(counts) / max(1, len(counts))))
        still = []
        for i, blocks in enumerate(counts):
            if blocks > 2 or i >= len(names_s) or i >= len(names_o):
                continue
            a = strokes(zs.read(names_s[i]))
            b = strokes(zo.read(names_o[i]))
            if a is None or b is None:
                continue
            kept = b / max(1, a)
            if b >= args.min_strokes and kept >= args.min_kept:
                still.append((i + 1, blocks, a, b, kept))
        low = sum(1 for c in counts if c <= 2)
        total_low += low
        total_still += len(still)
        print("  pages with <=2 blocks: %d   of those still carrying text: %d"
              % (low, len(still)))
        for page, blocks, a, b, kept in still:
            print("     page %-4d blocks=%-2d src_strokes=%-6d out_strokes=%-6d kept=%.2f"
                  % (page, blocks, a, b, kept))
        zs.close()
        zo.close()

    print("\n" + "=" * 70)
    print("TOTAL: %d low-block pages, %d of them still carrying text"
          % (total_low, total_still))
    print("(a low-block page that was NOT translated will show up here; a genuinely")
    print(" blank/credits page will not, because it has few strokes in the source too)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
