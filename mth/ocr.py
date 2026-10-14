import http.client
import json
import os
import sys
import time
import urllib.request

from . import providers
from .schema import new_block

_MANGA_OCR = None

# Vendored comic-text-detector. Upstream ships no package metadata, so it is
# used from a directory (the image vendors it to /opt/comic-text-detector); a
# pip-installed package is accepted as a fallback.
_DETECTOR_DIRS = (
    os.environ.get("COMIC_TEXT_DETECTOR_DIR", ""),
    "/opt/comic-text-detector",
)


def _text_detector_class():
    for directory in _DETECTOR_DIRS:
        if directory and os.path.isdir(directory) and directory not in sys.path:
            # append, never prepend: the checkout has generic top-level names
            # (`utils`, `models`) that must not shadow site-packages
            sys.path.append(directory)
    errors = []
    for module_name in ("inference", "comic_text_detector.inference"):
        try:
            module = __import__(module_name, fromlist=["TextDetector"])
            return module.TextDetector
        except Exception as exc:  # noqa: BLE001
            errors.append("%s: %s" % (module_name, exc))
    raise RuntimeError("comic-text-detector not importable (%s)" % "; ".join(errors))


def _detector_weights():
    override = os.environ.get("COMIC_TEXT_DETECTOR_WEIGHTS", "")
    for directory in ("",) + _DETECTOR_DIRS:
        candidate = (override if directory == "" else
                     os.path.join(directory, "data", "comictextdetector.pt"))
        if candidate and os.path.exists(candidate):
            return candidate
    raise RuntimeError(
        "comictextdetector.pt not found; set COMIC_TEXT_DETECTOR_DIR to a "
        "comic-text-detector checkout")


def detect_boxes(image, device="cpu"):
    """Text-line boxes from comic-text-detector (returns [(x0, y0, x1, y1), ...])."""
    import numpy as np
    detector = _text_detector_class()(model_path=_detector_weights(),
                                      input_size=1024, device=device, act="leaky")
    arr = np.array(image.convert("RGB"))[:, :, ::-1]
    _mask, _mask_refined, blocks = detector(arr)
    boxes = []
    for block in blocks:
        for line in getattr(block, "lines", None) or []:
            xs = [point[0] for point in line]
            ys = [point[1] for point in line]
            boxes.append([int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))])
    return boxes


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
    timeout = int(cfg.get("timeout", 300))

    # The host's port forward drops non-streaming requests at random (a plain
    # /prepare carries no bytes until it finishes), so retry instead of losing
    # the whole run on one dropped connection.
    attempts = max(1, int(cfg.get("retries", 3)))
    data = None
    last = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except (urllib.error.URLError, OSError, http.client.HTTPException) as err:
            last = err
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    if data is None:
        raise RuntimeError("OCR /prepare failed after %d attempts (%s): %s" % (
            attempts, url, last))

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
    image = Image.open(image_path).convert("RGB")
    try:
        boxes = detect_boxes(image, device=cfg.get("ocr_device", "cpu"))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "manga_ocr backend needs comic-text-detector for boxes: %s\n"
            "point COMIC_TEXT_DETECTOR_DIR at a comic-text-detector checkout" % exc
        )
    texts = recognize_manga_ocr(image, boxes)
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
