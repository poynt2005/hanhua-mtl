import math

from PIL import Image, ImageDraw, ImageFont

from . import inpaint


def _clamp(value, low, high):
    return max(low, min(high, value))


def _load(font_path, size):
    return ImageFont.truetype(font_path, size)


def _wrap(font, text, max_w):
    lines = []
    cur = ""
    for ch in text:
        if ch == "\n":
            lines.append(cur)
            cur = ""
            continue
        if font.getlength(cur + ch) <= max_w or not cur:
            cur += ch
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


def _fit_horizontal(draw, text, box_w, box_h, font_path):
    line_gap = 1.18
    best = None
    top = max(8, int(min(box_w, box_h)))
    for size in range(top, 7, -1):
        font = _load(font_path, size)
        lines = _wrap(font, text, box_w)
        need_h = len(lines) * size * line_gap
        if need_h <= box_h:
            best = (size, font, lines)
            break
    if best is None:
        size = 8
        font = _load(font_path, size)
        best = (size, font, _wrap(font, text, box_w))
    return best


def _fit_vertical(draw, text, box_w, box_h, font_path):
    col_gap = 1.12
    char_gap = 1.06
    n = len(text)
    best = None
    top = max(8, int(min(box_w, box_h)))
    for size in range(top, 7, -1):
        char_h = size * char_gap
        per_col = max(1, int(box_h // char_h))
        cols = math.ceil(n / per_col)
        need_w = cols * size * col_gap
        if need_w <= box_w:
            best = (size, per_col)
            break
    if best is None:
        size = 8
        best = (size, max(1, int(box_h // (size * char_gap))))
    return best


def _pick_color(image, box):
    region = image.crop((box[0], box[1], box[2], box[3]))
    small = region.resize((1, 1), Image.BOX).getpixel((0, 0))
    if isinstance(small, tuple):
        r, g, b = small[:3]
    else:
        r = g = b = small
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return (255, 255, 255) if lum < 120 else (0, 0, 0)


def _draw_horizontal(draw, text, box, font_path, color, stroke):
    x0, y0, x1, y1 = box
    box_w, box_h = x1 - x0, y1 - y0
    size, font, lines = _fit_horizontal(draw, text, box_w, box_h, font_path)
    line_gap = 1.18
    total_h = len(lines) * size * line_gap
    y = y0 + (box_h - total_h) / 2.0
    stroke_fill = tuple(255 - c for c in color)
    for line in lines:
        line_w = font.getlength(line)
        x = x0 + (box_w - line_w) / 2.0
        draw.text((x, y), line, font=font, fill=color,
                  stroke_width=stroke, stroke_fill=stroke_fill)
        y += size * line_gap


def _draw_vertical(draw, text, box, font_path, color, stroke):
    x0, y0, x1, y1 = box
    box_w, box_h = x1 - x0, y1 - y0
    size, per_col = _fit_vertical(draw, text, box_w, box_h, font_path)
    font = _load(font_path, size)
    col_gap = 1.12
    char_gap = 1.06
    cols = math.ceil(len(text) / per_col)
    cell_w = size * col_gap
    left = x0 + (box_w - cols * cell_w) / 2.0
    stroke_fill = tuple(255 - c for c in color)
    for col in range(cols):
        chunk = text[col * per_col:(col + 1) * per_col]
        col_x = left + (cols - 1 - col) * cell_w
        total_h = len(chunk) * size * char_gap
        y = y0 + (box_h - total_h) / 2.0
        for ch in chunk:
            cx = col_x + (cell_w - size) / 2.0
            draw.text((cx, y), ch, font=font, fill=color,
                      stroke_width=stroke, stroke_fill=stroke_fill)
            y += size * char_gap


def render_page(image, page, cfg, font_path, erase=True):
    boxes = []
    for block in page["blocks"]:
        if cfg.get("keep_sfx", True) and block.get("type") == "sfx":
            continue
        if not block.get("ocr"):
            continue
        boxes.append(block["bbox"])
    canvas = inpaint.erase(image, boxes, cfg) if erase else image.copy()
    draw = ImageDraw.Draw(canvas)
    stroke = int(cfg.get("stroke", 0))
    for block in page["blocks"]:
        if cfg.get("keep_sfx", True) and block.get("type") == "sfx":
            continue
        text = (block.get("translation") or "").strip()
        if not text:
            continue
        box = block["bbox"]
        color = _pick_color(canvas, box)
        direction = block.get("direction", "auto")
        if direction == "horizontal":
            _draw_horizontal(draw, text, box, font_path, color, stroke)
        elif direction == "vertical":
            _draw_vertical(draw, text, box, font_path, color, stroke)
        elif len(text) >= 8 and (box[3] - box[1]) > (box[2] - box[0]) * 1.3:
            _draw_vertical(draw, text, box, font_path, color, stroke)
        else:
            _draw_horizontal(draw, text, box, font_path, color, stroke)
    return canvas
