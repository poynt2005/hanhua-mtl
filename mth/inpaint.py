from PIL import Image, ImageDraw, ImageFilter

_BLUR_BOX = 9
_LAMA = None


def _clamp(value, low, high):
    return max(low, min(high, value))


def _expand(box, pad, width, height):
    x0, y0, x1, y1 = box
    return (
        _clamp(x0 - pad, 0, width),
        _clamp(y0 - pad, 0, height),
        _clamp(x1 + pad, 0, width),
        _clamp(y1 + pad, 0, height),
    )


def _border_mean(image, box):
    w, h = image.size
    x0, y0, x1, y1 = box
    strips = [
        (x0 - 3, y0, x0, y1),
        (x1, y0, x1 + 3, y1),
        (x0, y0 - 3, x1, y0),
        (x0, y1, x1, y1 + 3),
    ]
    samples = []
    for sx0, sy0, sx1, sy1 in strips:
        sx0, sy0 = _clamp(sx0, 0, w), _clamp(sy0, 0, h)
        sx1, sy1 = _clamp(sx1, 0, w), _clamp(sy1, 0, h)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        samples.append(image.crop((sx0, sy0, sx1, sy1)).resize((1, 1), Image.BOX).getpixel((0, 0)))
    if not samples:
        return (255, 255, 255)
    channels = list(zip(*samples))
    return tuple(int(sorted(c)[len(c) // 2]) for c in channels)


def _fill_method(image, boxes, pad):
    w, h = image.size
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for box in boxes:
        region = _expand(box, pad, w, h)
        color = _border_mean(image, region)
        draw.rectangle(region, fill=color)
    return out


def _blur_method(image, boxes, pad):
    w, h = image.size
    out = image.copy()
    blurred = image.filter(ImageFilter.GaussianBlur(_BLUR_BOX))
    for box in boxes:
        region = _expand(box, pad, w, h)
        crop = blurred.crop(region)
        out.paste(crop, (region[0], region[1]))
    return out


def _opencv_method(image, boxes, pad):
    import cv2
    import numpy as np
    w, h = image.size
    arr = np.array(image.convert("RGB"))[:, :, ::-1].copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        x0, y0, x1, y1 = _expand(box, pad, w, h)
        mask[y0:y1, x0:x1] = 255
    result = cv2.inpaint(arr, mask, 3, cv2.INPAINT_TELEA)
    return Image.fromarray(result[:, :, ::-1])


def _lama_method(image, boxes, pad):
    import numpy as np
    from simple_lama_inpainting import SimpleLama
    global _LAMA
    if _LAMA is None:
        _LAMA = SimpleLama()
    w, h = image.size
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        x0, y0, x1, y1 = _expand(box, pad, w, h)
        mask[y0:y1, x0:x1] = 255
    return _LAMA(image.convert("RGB"), mask).convert("RGB")


def _textmask_method(image, boxes, pad, dilate=7):
    """Erase only the text strokes (Otsu inside each box) via cv2.inpaint.

    Unlike a solid rectangle fill, this preserves bubble borders and artwork
    that fall inside the box, and removes antialiased glyph edges by dilating
    the text mask before inpainting.
    """
    import cv2
    import numpy as np
    arr = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        x0, y0, x1, y1 = _expand(box, pad, w, h)
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        roi = gray[y0:y1, x0:x1]
        _, inv = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, fwd = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw = inv if cv2.countNonZero(inv) <= cv2.countNonZero(fwd) else fwd
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        bw = cv2.dilate(bw, kernel, iterations=1)
        mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], bw)
    out = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


def erase(image, boxes, cfg):
    method = cfg.get("inpaint", "fill")
    pad = int(cfg.get("padding", 3))
    if not boxes:
        return image
    if method in ("textmask", "auto"):
        try:
            return _textmask_method(image, boxes, pad, int(cfg.get("dilate", 7)))
        except Exception as exc:
            print("[warn] textmask unavailable (%s); falling back to fill" % exc)
            return _fill_method(image, boxes, pad)
    if method == "lama":
        try:
            return _lama_method(image, boxes, pad)
        except Exception as exc:
            print("[warn] lama unavailable (%s); falling back to blur" % exc)
            return _blur_method(image, boxes, pad)
    if method == "opencv":
        return _opencv_method(image, boxes, pad)
    if method == "blur":
        return _blur_method(image, boxes, pad)
    return _fill_method(image, boxes, pad)
