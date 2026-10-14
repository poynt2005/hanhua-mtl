"""Contact sheets of the low-block pages, taken from the DELIVERED archives.

If a page still shows the source language, detection missed it and nothing was
translated there. This is the human-eye version of tools/audit_detection.py: it
produces one PNG per volume, each cell labelled `page N (M blocks)`.

    python tools/audit_sheets.py --dir . --out-dir . --cols 6 --width 300
"""
import argparse
import io
import json
import math
import os
import sys
import zipfile

from PIL import Image, ImageDraw

PAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")


def build(folder, name, out_dir, cols, thumb_w):
    stem = name[:-len("_MTL_zhTW.zip.project.json")]
    out_zip = os.path.join(folder, stem + "_MTL_zhTW.zip")
    if not os.path.exists(out_zip):
        return None
    data = json.load(open(os.path.join(folder, name), encoding="utf-8"))
    counts = [len(p.get("blocks") or []) for p in (data.get("pages") or [])]
    suspect = [i for i, c in enumerate(counts) if c <= 2]
    if not suspect:
        return None

    with zipfile.ZipFile(out_zip) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(PAGE_EXT)]
        thumbs = []
        for idx in suspect:
            if idx >= len(names):
                continue
            im = Image.open(io.BytesIO(zf.read(names[idx]))).convert("RGB")
            h = int(im.height * thumb_w / im.width)
            thumbs.append((idx + 1, counts[idx], im.resize((thumb_w, h), Image.LANCZOS)))
    if not thumbs:
        return None

    rows = math.ceil(len(thumbs) / cols)
    cell_h = max(t[2].height for t in thumbs)
    pad, label_h = 8, 22
    sheet = Image.new("RGB", (cols * (thumb_w + pad) + pad,
                              rows * (cell_h + label_h + pad) + pad), (30, 30, 34))
    draw = ImageDraw.Draw(sheet)
    for n, (page, blocks, thumb) in enumerate(thumbs):
        x = pad + (n % cols) * (thumb_w + pad)
        y = pad + (n // cols) * (cell_h + label_h + pad)
        sheet.paste(thumb, (x, y))
        draw.rectangle([x - 1, y - 1, x + thumb_w, y + cell_h], outline=(120, 120, 130))
        draw.text((x + 4, y + cell_h + 4), "page %d  (%d blocks)" % (page, blocks),
                  fill=(255, 220, 120))

    safe = "".join(c if c.isalnum() else "_" for c in stem[:26])
    out = os.path.join(out_dir, "audit_%s.png" % safe)
    sheet.save(out)
    print("wrote %s  (%d suspect pages, %dx%d)" % (out, len(thumbs), sheet.width, sheet.height))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=".", help="folder holding the archives")
    ap.add_argument("--out-dir", default=".", help="where to write the PNGs")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--width", type=int, default=300, help="thumbnail width in px")
    args = ap.parse_args()

    made = 0
    for name in sorted(os.listdir(args.dir)):
        if name.endswith("_MTL_zhTW.zip.project.json") and not name.startswith("pilot"):
            if build(args.dir, name, args.out_dir, args.cols, args.width):
                made += 1
    print("\ncontact sheets written: %d" % made)
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
