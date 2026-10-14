"""Manga OCR + inpaint service for the hanhua pipeline (run on a GPU box).

General design (works across art styles, not tuned to one book):
  detect  : comic-text-detector (boxes + segmentation mask) if available,
            else EasyOCR boxes
  OCR     : manga-ocr (Japanese, manga-specialised)
  inpaint : LaMa (neural, mask-driven) -> returns a clean page

POST /prepare (multipart field "file") returns:
  {"blocks": [{"bbox":[x0,y0,x1,y1], "text":..., "direction":..., "type":...}],
   "cleaned": "<base64 PNG of the page with text erased>"}
POST /ocr  -> just the blocks (kept for backward compatibility)
GET  /health
"""
import base64
import faulthandler
import io
import os
import sys

import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image

app = FastAPI(title="manga ocr+inpaint server")

TOKEN = os.environ.get("OCR_TOKEN", "")
DILATE = int(os.environ.get("INPAINT_DILATE", "9"))

# Vendored comic-text-detector checkout (see deploy/Dockerfile). Upstream ships
# no package metadata, so it can only be used as a plain directory of modules.
DETECTOR_DIR = os.environ.get("COMIC_TEXT_DETECTOR_DIR", "/opt/comic-text-detector")

# Detection passes to union. A single pass misses whole classes of text:
#   asis   - normal black-on-light text
#   rot90  - vertical lines become horizontal (the detector is far better at
#            horizontal text; manga-image-translator has the same idea as
#            `det_auto_rotate`)
#   rot270 - the other direction, in case the page is mirrored/rotated
#   invert - white text on dark backgrounds
# Measured motivation: a cv2 audit found ~32-74 of 1034 delivered pages still
# carried untranslated text because a single pass had missed it.
DET_VARIANTS = [v.strip() for v in os.environ.get(
    "DET_VARIANTS", "asis,rot90,rot270,invert").split(",") if v.strip()]
DET_SIZE = int(os.environ.get("DET_SIZE", "1280"))
# Only escalate to the extra passes when the plain pass found suspiciously
# little text - the extra passes cost ~4x and most pages do not need them.
DET_ESCALATE_BELOW = int(os.environ.get("DET_ESCALATE_BELOW", "5"))

# Every log line also lands in this file, and GET /logs serves it over HTTP. That
# is deliberate: a native crash kills the process, and without this the cause is
# unrecoverable (no SSH into the container, stderr is redirected elsewhere).
LOG_PATH = os.environ.get("OCR_LOG", "/root/ocr.log")


def log(msg):
    text = "[ocr] " + str(msg)
    print(text, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(text + "\n")
    except OSError:
        pass


# A hard crash (segfault/abort) otherwise kills the connection with no
# explanation; faulthandler dumps a traceback into the same log file.
try:
    _fh = open(LOG_PATH, "a", buffering=1, encoding="utf-8")
    faulthandler.enable(_fh)
except OSError:
    faulthandler.enable()


def vram():
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = total - free
            return "vram %.1f/%.1f GB used" % (used / 2 ** 30, total / 2 ** 30)
        return "vram no-cuda"
    except Exception:  # noqa: BLE001
        return "vram n/a"


def _excepthook(exc_type, exc, tb):
    import traceback
    log("UNCAUGHT: " + "".join(traceback.format_exception(exc_type, exc, tb)))


sys.excepthook = _excepthook

_reader = None
_mocr = None
_detector = None
_lama = None


def get_reader():
    global _reader
    if _reader is None:
        import easyocr
        import torch
        _reader = easyocr.Reader(["ja"], gpu=torch.cuda.is_available())
    return _reader


def get_mocr():
    global _mocr
    if _mocr is None:
        from manga_ocr import MangaOcr
        _mocr = MangaOcr()
    return _mocr


def _pick_device():
    """Prefer CUDA only when a GPU is actually visible.

    Hard-coding "cuda" made the detector fail to load anywhere without a GPU
    (including the build-time smoke test), which silently downgraded detection
    to EasyOCR. `DEVICE` still overrides when set.
    """
    import torch
    return os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def _import_text_detector():
    """Return the TextDetector class from the vendored comic-text-detector.

    The upstream project ships no package metadata (no setup.py / pyproject.toml)
    and uses top-level imports, so the build clones it to a fixed directory and
    that directory is APPENDED to sys.path — appended, never prepended, because
    the checkout has generic top-level names (`utils`, `models`) that must not
    shadow site-packages. A pip-installed package is accepted as a fallback.
    """
    if DETECTOR_DIR and os.path.isdir(DETECTOR_DIR) and DETECTOR_DIR not in sys.path:
        sys.path.append(DETECTOR_DIR)
    errors = []
    for module_name in ("inference", "comic_text_detector.inference"):
        try:
            module = __import__(module_name, fromlist=["TextDetector"])
            return module.TextDetector
        except Exception as exc:  # noqa: BLE001
            errors.append("%s: %s" % (module_name, exc))
    raise ImportError("; ".join(errors))


def _detector_weights():
    for candidate in (
        os.environ.get("COMIC_TEXT_DETECTOR_WEIGHTS", ""),
        os.path.join(DETECTOR_DIR, "data", "comictextdetector.pt"),
    ):
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def get_detector():
    """comic-text-detector gives boxes AND a per-pixel text mask (general)."""
    global _detector
    if _detector is None:
        try:
            weights = _detector_weights()
            if not weights:
                raise FileNotFoundError(
                    "comictextdetector.pt not found (looked under %s)" % DETECTOR_DIR)
            detector_cls = _import_text_detector()
            _detector = detector_cls(model_path=weights, input_size=DET_SIZE,
                                     device=_pick_device(), act="leaky")
        except Exception as exc:  # noqa: BLE001
            print("[warn] comic-text-detector unavailable: %s" % exc)
            _detector = False
    return _detector or None


def get_lama():
    global _lama
    if _lama is None:
        from simple_lama_inpainting import SimpleLama
        _lama = SimpleLama()
    return _lama


def _line_boxes(blocks):
    boxes = []
    for block in blocks:
        for line in getattr(block, "lines", None) or []:
            xs = [point[0] for point in line]
            ys = [point[1] for point in line]
            boxes.append([int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))])
    return boxes


def _rot90_box_to_base(box, base_w):
    """Map a box detected on np.rot90(base, 1) back to base coordinates.

    rot90(m, 1) sends base (x, y) to (x' = y, y' = base_w - 1 - x), so the
    inverse of a box is [base_w-1-y1, x0, base_w-1-y0, x1].
    """
    x0, y0, x1, y1 = box
    return [base_w - 1 - y1, x0, base_w - 1 - y0, x1]


def _rot270_box_to_base(box, base_h):
    """Map a box detected on np.rot90(base, 3) back to base coordinates.

    rot90(m, 3) sends base (x, y) to (x' = base_h - 1 - y, y' = x), so the
    inverse of a box is [y0, base_h-1-x1, y1, base_h-1-x0].
    """
    x0, y0, x1, y1 = box
    return [y0, base_h - 1 - x1, y1, base_h - 1 - x0]


def _box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union else 0.0


def _dedup_boxes(boxes, iou_limit=0.4):
    """The passes overlap heavily; keep the largest of each cluster so the OCR
    stage does not do the same crop several times."""
    kept = []
    for box in sorted(boxes, key=_box_area, reverse=True):
        if _box_area(box) <= 0:
            continue
        if any(_iou(box, other) > iou_limit for other in kept):
            continue
        kept.append(box)
    return kept


def _run_variant(det, variant, base, width, height):
    """One detector pass. Returns (boxes_in_base_coords, mask_in_base_coords)."""
    if variant == "asis":
        arr, back, mask_back = base, None, None
    elif variant == "rot90":
        arr = np.rot90(base, 1).copy()
        back = lambda box: _rot90_box_to_base(box, width)   # noqa: E731
        mask_back = lambda m: np.rot90(m, -1)               # noqa: E731
    elif variant == "rot270":
        arr = np.rot90(base, 3).copy()
        back = lambda box: _rot270_box_to_base(box, height)  # noqa: E731
        mask_back = lambda m: np.rot90(m, 1)                 # noqa: E731
    elif variant == "invert":
        arr = (255 - base).astype(np.uint8)
        back, mask_back = None, None
    else:
        return [], None
    _mask, mask, blocks = det(arr)
    boxes = _line_boxes(blocks)
    if back is not None:
        boxes = [back(box) for box in boxes]
    mask = np.asarray(mask)
    if mask_back is not None and getattr(mask, "size", 0):
        mask = mask_back(mask)
    if getattr(mask, "size", 0):
        mask = mask[:height, :width]
    else:
        mask = None
    return boxes, mask


def detect(img):
    """Return (boxes, text_mask_or_None).

    The first pass is the plain one. Extra passes (rotated, inverted) are only
    run when that pass finds suspiciously little text - those passes recover
    vertical lines and white-on-dark text, but they cost 4x, and most pages do
    not need them. Measured motivation: ~32-74 of 1034 delivered pages still
    carried untranslated text because a single pass had missed it.
    """
    det = get_detector()
    if det is not None:
        try:
            base = np.array(img)[:, :, ::-1]          # BGR, as the detector wants
            height, width = base.shape[:2]
            order = [v for v in DET_VARIANTS]
            if "asis" in order:
                order.remove("asis")
                order.insert(0, "asis")

            all_boxes, mask_total, used = [], None, []
            for index, variant in enumerate(order):
                if index and len(all_boxes) >= DET_ESCALATE_BELOW:
                    break          # the first pass found enough; skip the rest
                try:
                    boxes, mask = _run_variant(det, variant, base, width, height)
                except Exception as exc:  # noqa: BLE001
                    log("detect[%s] failed: %s" % (variant, exc))
                    continue
                used.append(variant)
                log("detect[%s] -> %d boxes%s" % (
                    variant, len(boxes),
                    "  (escalating: too few)" if index == 0 and len(boxes) < DET_ESCALATE_BELOW else ""))
                all_boxes.extend(boxes)
                if mask is not None:
                    mask_total = mask if mask_total is None else np.maximum(mask_total, mask)

            merged = _dedup_boxes(all_boxes)
            log("detect: %d raw -> %d after dedup (passes used=%s)"
                % (len(all_boxes), len(merged), ",".join(used)))
            return merged, mask_total
        except Exception as exc:  # noqa: BLE001
            log("detector failed, falling back to easyocr: %s" % exc)
    results = get_reader().readtext(np.array(img), detail=1, paragraph=False)
    boxes = []
    for bbox, _t, _c in results:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        boxes.append([int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))])
    return boxes, None


def build_mask(img, boxes, seg_mask, dilate=DILATE):
    """Precise text mask: segmentation mask if available, else Otsu inside boxes.

    Using whole rectangles as the mask makes LaMa redraw bubble borders and any
    artwork inside the box; an Otsu text mask keeps the erase to text strokes.
    """
    import cv2
    gray = np.array(img.convert("L"))
    h, w = gray.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if seg_mask is not None and getattr(seg_mask, "size", 0):
        m = seg_mask
        m = (m > 0.5).astype(np.uint8) * 255 if m.max() <= 1 else (m > 127).astype(np.uint8) * 255
        mask = np.maximum(mask, m[:h, :w])
    else:
        for x0, y0, x1, y1 in boxes:
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            roi = gray[y0:y1, x0:x1]
            _, inv = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            _, fwd = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bw = inv if cv2.countNonZero(inv) <= cv2.countNonZero(fwd) else fwd
            mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], bw)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
    return cv2.dilate(mask, kernel, iterations=1)


def inpaint(img, mask):
    lama = get_lama()
    out = lama(img, mask).convert("RGB")
    if out.size != img.size:
        # LaMa pads to a multiple of 8; crop back to the original size.
        out = out.crop((0, 0, img.width, img.height))
    return out


def read_blocks(img, boxes):
    mocr = get_mocr()
    blocks = []
    for box in boxes:
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:
            continue
        text = mocr(img.crop(tuple(box))).strip()
        if not text:
            continue
        blocks.append({"bbox": box, "text": text, "direction": "auto", "type": "speech"})
    return blocks


def _auth(authorization):
    if TOKEN and authorization != "Bearer " + TOKEN:
        raise HTTPException(status_code=401, detail="bad token")


def _hf_mbps():
    try:
        with open("/root/hf_mbps.txt", "r", encoding="ascii") as handle:
            return int(handle.read().strip())
    except Exception:  # noqa: BLE001
        return None


@app.get("/health")
def health():
    return {"ok": True, "detector": bool(get_detector()), "inpaint": "lama",
            "hf_mbps": _hf_mbps(), "vram": vram()}


@app.get("/logs")
def logs(n: int = 200):
    """Tail of the service log. Survives a crash, so the cause can be fetched
    over HTTP even when the process died (no SSH needed)."""
    try:
        with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()[-max(1, min(int(n), 5000)):]
        return {"path": LOG_PATH, "lines": [line.rstrip("\n") for line in lines]}
    except OSError as exc:
        return {"path": LOG_PATH, "error": str(exc), "lines": []}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...), authorization: str = Header(default="")):
    _auth(authorization)
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    boxes, _seg = detect(img)
    return {"blocks": read_blocks(img, boxes)}


@app.post("/prepare")
async def prepare(file: UploadFile = File(...), authorization: str = Header(default="")):
    _auth(authorization)
    try:
        raw = await file.read()
        log("prepare: received %d bytes" % len(raw))
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        log("prepare: page %dx%d | %s" % (img.width, img.height, vram()))

        log("prepare: detect ...")
        boxes, seg = detect(img)
        log("prepare: detect -> %d boxes, seg=%s | %s" % (
            len(boxes), None if seg is None else tuple(seg.shape), vram()))

        log("prepare: ocr ...")
        blocks = read_blocks(img, boxes)
        log("prepare: ocr -> %d blocks | %s" % (len(blocks), vram()))

        cleaned_b64 = None
        try:
            log("prepare: mask ...")
            mask = build_mask(img, boxes, seg)
            log("prepare: inpaint ... (%d px masked) | %s" % (int((mask > 0).sum()), vram()))
            cleaned = inpaint(img, mask)
            buf = io.BytesIO()
            cleaned.save(buf, format="PNG")
            cleaned_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            log("prepare: done | %s" % vram())
        except Exception as exc:  # noqa: BLE001
            log("prepare: inpaint unavailable (%s); returning blocks only" % exc)
        return {"blocks": blocks, "cleaned": cleaned_b64}
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001
        import traceback
        detail = traceback.format_exc()
        log("prepare: FAILED\n" + detail)
        raise HTTPException(status_code=500, detail=detail[-2000:])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
