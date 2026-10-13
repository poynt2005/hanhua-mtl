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
import io
import os

import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image

app = FastAPI(title="manga ocr+inpaint server")

TOKEN = os.environ.get("OCR_TOKEN", "")
DILATE = int(os.environ.get("INPAINT_DILATE", "9"))

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


def get_detector():
    """comic-text-detector gives boxes AND a per-pixel text mask (general)."""
    global _detector
    if _detector is None:
        try:
            import comic_text_detector
            from comic_text_detector.inference import TextDetector
            model_path = os.path.join(os.path.dirname(comic_text_detector.__file__),
                                      "data", "comictextdetector.pt")
            _detector = TextDetector(model_path=model_path, input_size=1024,
                                     device=os.environ.get("DEVICE", "cuda"), act="leaky")
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


def detect(img):
    """Return (boxes, mask_or_None)."""
    arr = np.array(img)[:, :, ::-1]
    det = get_detector()
    if det is not None:
        try:
            mask, groups, _ = det(arr)
            boxes = []
            for group in groups:
                for line in group:
                    xs = [p[0] for p in line]
                    ys = [p[1] for p in line]
                    boxes.append([int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))])
            return boxes, np.asarray(mask)
        except Exception as exc:  # noqa: BLE001
            print("[warn] detector failed, falling back to easyocr: %s" % exc)
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
            "hf_mbps": _hf_mbps()}


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
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        boxes, seg = detect(img)
        blocks = read_blocks(img, boxes)
        cleaned_b64 = None
        try:
            mask = build_mask(img, boxes, seg)
            cleaned = inpaint(img, mask)
            buf = io.BytesIO()
            cleaned.save(buf, format="PNG")
            cleaned_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as exc:  # noqa: BLE001
            print("[warn] inpaint unavailable (%s); returning blocks only" % exc)
        return {"blocks": blocks, "cleaned": cleaned_b64}
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001
        import traceback
        raise HTTPException(status_code=500, detail=traceback.format_exc()[-2000:])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
