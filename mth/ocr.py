import json
import os
import urllib.request

from . import providers
from .schema import new_block

_MANGA_OCR = None


def _load_manga_ocr():
    global _MANGA_OCR
    if _MANGA_OCR is None:
        from manga_ocr import MangaOcr
        _MANGA_OCR = MangaOcr()
    return _MANGA_OCR


def recognize_manga_ocr(image, boxes):
    from PIL import Image
    mocr = _load_manga_ocr()
    texts = []
    for box in boxes:
        crop = image.crop((box[0], box[1], box[2], box[3]))
        try:
            texts.append(mocr(crop).strip())
        except Exception:
            texts.append("")
    return texts


def _vision(image_path, width, height, cfg, client):
    return providers.vision_extract(client, cfg["vision_model"], image_path, width, height)


def _items_to_blocks(data):
    if isinstance(data, dict):
        data = data.get("blocks") or data.get("regions") or data.get("data") or []
    blocks = []
    for item in data:
        box = item.get("bbox") or item.get("box") or item.get("rect")
        if not box or len(box) != 4:
            continue
        blocks.append({
            "bbox": [int(round(float(v))) for v in box],
            "ocr": (item.get("text") or item.get("ocr") or "").strip(),
            "direction": item.get("direction", "auto"),
            "type": item.get("type", "speech"),
        })
    return blocks


def _sidecar(image_path, cfg):
    side = os.path.splitext(image_path)[0] + ".json"
    if not os.path.exists(side):
        raise FileNotFoundError("sidecar not found: %s" % side)
    with open(side, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return _items_to_blocks(data)


def _prepare_url(cfg):
    if cfg.get("ocr_prepare_url"):
        return cfg["ocr_prepare_url"]
    url = (cfg.get("ocr_url") or "").rstrip("/")
    if not url:
        raise ValueError("--ocr http needs config ocr_url (your self-hosted OCR endpoint)")
    if url.endswith("/prepare"):
        return url
    if url.endswith("/ocr"):
        return url[:-4] + "/prepare"
    return url + "/prepare"


def _http(image_path, cfg):
    url = _prepare_url(cfg)
    boundary = "----mthBoundary8f2a1c"
    with open(image_path, "rb") as handle:
        raw = handle.read()
    filename = os.path.basename(image_path)
    body = b""
    body += ("--%s\r\n" % boundary).encode()
    body += ('Content-Disposition: form-data; name="file"; filename="%s"\r\n' % filename).encode()
    body += b"Content-Type: application/octet-stream\r\n\r\n" + raw + b"\r\n"
    body += ("--%s--\r\n" % boundary).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "multipart/form-data; boundary=%s" % boundary)
    key = cfg.get("ocr_api_key")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=int(cfg.get("timeout", 300))) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    cleaned = data.get("cleaned") if isinstance(data, dict) else None
    if cleaned:
        import base64
        clean_path = os.path.splitext(image_path)[0] + ".clean.png"
        with open(clean_path, "wb") as handle:
            handle.write(base64.b64decode(cleaned))
    return _items_to_blocks(data)


def _opencv(image_path, width, height, cfg):
    import cv2
    import numpy as np
    from PIL import Image
    bgr = cv2.imread(image_path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dil = cv2.dilate(bw, kernel, iterations=2)
    count, _, stats, _ = cv2.connectedComponentsWithStats(dil, 8)
    boxes = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < 40 or w < 6 or h < 6:
            continue
        boxes.append([int(x), int(y), int(x + w), int(y + h)])
    blocks = [{"bbox": b, "ocr": "", "direction": "auto", "type": "speech"} for b in boxes]
    try:
        pil = Image.open(image_path).convert("RGB")
        texts = recognize_manga_ocr(pil, boxes)
        for block, text in zip(blocks, texts):
            block["ocr"] = text
    except Exception as exc:
        print("[warn] OCR recognition unavailable (%s); boxes only" % exc)
    return blocks


def _manga_ocr(image_path, width, height, cfg):
    from PIL import Image
    try:
        from comic_text_detector.inference import TextDetector
        import comic_text_detector
    except Exception as exc:
        raise RuntimeError(
            "manga_ocr backend needs comic-text-detector for boxes: %s\n"
            "install with: pip install manga-ocr comic-text-detector" % exc
        )
    import numpy as np
    model_path = os.path.join(os.path.dirname(comic_text_detector.__file__), "data", "comictextdetector.pt")
    detector = TextDetector(model_path=model_path, input_size=1024, device="cpu", act="leaky")
    bgr = np.array(Image.open(image_path).convert("RGB"))[:, :, ::-1]
    mask, blocks_raw, _ = detector(bgr)
    boxes = []
    for blk in blocks_raw:
        for line in blk:
            xs = [pt[0] for pt in line]
            ys = [pt[1] for pt in line]
            boxes.append([int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))])
    texts = recognize_manga_ocr(Image.open(image_path).convert("RGB"), boxes)
    return [{
        "bbox": b,
        "ocr": t,
        "direction": "auto",
        "type": "speech",
    } for b, t in zip(boxes, texts)]


def _mock(image_path, width, height, cfg):
    box = [int(width * 0.12), int(height * 0.08), int(width * 0.88), int(height * 0.22)]
    return [{"bbox": box, "ocr": "テスト用のテキストです", "direction": "vertical", "type": "speech"}]


def extract_page(image_path, width, height, backend, cfg, client=None):
    if backend == "vision":
        return _vision(image_path, width, height, cfg, client)
    if backend == "sidecar":
        return _sidecar(image_path, cfg)
    if backend == "http":
        return _http(image_path, cfg)
    if backend == "opencv":
        return _opencv(image_path, width, height, cfg)
    if backend == "manga_ocr":
        return _manga_ocr(image_path, width, height, cfg)
    if backend == "mock":
        return _mock(image_path, width, height, cfg)
    raise ValueError("unknown ocr backend: %s" % backend)
