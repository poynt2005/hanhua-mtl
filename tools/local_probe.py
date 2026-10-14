"""Free, torch-free probe of the vendored detector's non-model code paths.

The heavy parts (the detector forward) need torch, but a lot of the pipeline is
plain numpy/opencv: the non-contiguous BGR view handed to cv2, `letterbox`, the
`refine_mask` post-processing and `build_mask`. Those can be exercised locally
on a REAL page for nothing, which is how a suspected crash was ruled out.

    git clone --depth 1 https://github.com/dmMaze/comic-text-detector.git /tmp/ctd
    python tools/local_probe.py --vendor /tmp/ctd --dir . --pages 250

Needs: numpy, opencv-python(-headless), shapely, pillow.
"""
import argparse
import io
import os
import sys
import zipfile

import cv2
import numpy as np
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


def build_mask(image, boxes, seg_mask, dilate=9):
    gray = np.array(image.convert("L"))
    hh, ww = gray.shape[:2]
    mask = np.zeros((hh, ww), dtype=np.uint8)
    if seg_mask is not None and getattr(seg_mask, "size", 0):
        m = seg_mask
        m = (m > 0.5).astype(np.uint8) * 255 if m.max() <= 1 else (m > 127).astype(np.uint8) * 255
        mask = np.maximum(mask, m[:hh, :ww])
    else:
        for x0, y0, x1, y1 in boxes:
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(ww, x1), min(hh, y1)
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            roi = gray[y0:y1, x0:x1]
            _, inv = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            _, fwd = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bw = inv if cv2.countNonZero(inv) <= cv2.countNonZero(fwd) else fwd
            mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], bw)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
    return cv2.dilate(mask, kernel, iterations=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vendor", required=True, help="comic-text-detector checkout")
    ap.add_argument("--dir", default=".", help="folder with an input archive")
    ap.add_argument("--pages", type=int, default=250, help="page count to find")
    args = ap.parse_args()

    src = find_input(os.path.abspath(args.dir), args.pages)
    if not src:
        print("no archive with %d pages in %s" % (args.pages, args.dir))
        return 2
    with zipfile.ZipFile(src) as zf:
        name = pages(zf)[0]
        raw = zf.read(name)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    print("page: %s %s" % (name, img.size))

    arr = np.array(img)[:, :, ::-1]
    print("1) BGR view: C_CONTIGUOUS=%s" % arr.flags["C_CONTIGUOUS"])
    try:
        cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        cv2.resize(arr, (256, 256))
        print("   cv2 on the view: OK")
    except Exception as exc:  # noqa: BLE001
        print("   cv2 FAILED: %s" % exc)

    sys.path.insert(0, os.path.abspath(args.vendor))
    try:
        from utils.imgproc_utils import letterbox
        lb, ratio, _ = letterbox(arr, new_shape=(1024, 1024), auto=False, stride=64)
        print("2) letterbox OK -> %s" % (lb.shape,))
    except Exception as exc:  # noqa: BLE001
        print("2) letterbox FAILED: %s" % exc)

    try:
        from utils.textblock import TextBlock
        from utils.textmask import refine_mask, REFINEMASK_INPAINT
        h, w = arr.shape[:2]
        lines = [[[int(w * .18), int(h * .10)], [int(w * .78), int(h * .11)],
                  [int(w * .78), int(h * .20)], [int(w * .18), int(h * .19)]]]
        blk = TextBlock(xyxy=[int(w * .15), int(h * .08), int(w * .80), int(h * .35)],
                        lines=lines, language="ja")
        pred = (np.random.default_rng(0).random((h, w)) * 255).astype(np.uint8)
        refined = refine_mask(arr, pred, [blk], refine_mode=REFINEMASK_INPAINT)
        print("3) refine_mask OK -> dtype=%s shape=%s max=%s"
              % (refined.dtype, refined.shape, refined.max()))
        m = build_mask(img, [[int(w * .15), int(h * .08), int(w * .80), int(h * .35)]], refined)
        print("4) build_mask OK -> %s nonzero=%d" % (m.shape, int((m > 0).sum())))
    except Exception as exc:  # noqa: BLE001
        import traceback
        print("3/4) FAILED:")
        traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
