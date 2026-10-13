import glob
import os

from PIL import Image

from . import ocr, providers, render
from .schema import new_block, new_page, new_project


def list_images(input_dir, pattern):
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        for alt in ("*.jpg", "*.jpeg", "*.webp", "*.bmp"):
            files.extend(glob.glob(os.path.join(input_dir, alt)))
        files = sorted(files)
    return files


def reading_order(blocks):
    if not blocks:
        return blocks
    heights = [b["bbox"][3] - b["bbox"][1] for b in blocks]
    row = max(20, int(sum(heights) / len(heights) * 0.6))
    return sorted(blocks, key=lambda b: (round(b["bbox"][1] / row), -b["bbox"][0]))


def do_extract(input_dir, cfg, backend, client=None):
    project = new_project(cfg["source_lang"], cfg["target_lang"], os.path.basename(os.path.abspath(input_dir)))
    files = list_images(input_dir, cfg.get("pattern", "*.png"))
    if not files:
        raise FileNotFoundError("no images found in %s" % input_dir)
    for index, path in enumerate(files, start=1):
        with Image.open(path) as img:
            width, height = img.size
        raw = ocr.extract_page(path, width, height, backend, cfg, client)
        raw = reading_order(raw)
        blocks = []
        for i, item in enumerate(raw):
            x0, y0, x1, y1 = item["bbox"]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(width, x1), min(height, y1)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            block = new_block(
                "p%03d-b%02d" % (index, i),
                [x0, y0, x1, y1],
                item.get("ocr", ""),
                item.get("direction", "auto"),
                item.get("type", "speech"),
            )
            blocks.append(block)
        project["pages"].append(new_page(index, os.path.basename(path), width, height, blocks))
    return project


def _convert_tw(texts):
    try:
        import opencc
        conv = opencc.OpenCC("s2twp")
        return [conv.convert(t) for t in texts]
    except Exception:
        return texts


def do_translate(project, cfg, client):
    glossary_file = cfg.get("glossary_file")
    if glossary_file:
        from . import terms
        _, flat_map = terms.load_glossary(glossary_file)
        terms.apply_to_project(project, flat_map)
    pairs = []
    for page in project["pages"]:
        for block in page["blocks"]:
            if block.get("ocr"):
                pairs.append({"id": block["id"], "ja": block["ocr"]})
    if not pairs:
        return project
    profile = cfg.get("profile", "openai")
    system = cfg.get("system_prompt") or None
    chunk_size = int(cfg.get("translate_chunk", 3000))
    max_lines = int(cfg.get("translate_max_lines", 40))
    mapping = {}
    history = []

    def run(current):
        result = providers.translate_batch(
            client, cfg["text_model"], current, project.get("glossary"),
            cfg["target_lang"], profile, system, history[-10:] or None)
        for item in current:
            text = result.get(item["id"])
            if text:
                history.append(text)
        return result

    chunk = []
    count = 0
    for pair in pairs:
        length = len(pair["ja"])
        if chunk and (count + length > chunk_size or len(chunk) >= max_lines):
            mapping.update(run(chunk))
            chunk = []
            count = 0
        chunk.append(pair)
        count += length
    if chunk:
        mapping.update(run(chunk))
    if cfg.get("convert_tw", True):
        ids = list(mapping.keys())
        converted = _convert_tw([mapping[i] for i in ids])
        mapping = dict(zip(ids, converted))
    for page in project["pages"]:
        for block in page["blocks"]:
            if block["id"] in mapping:
                block["translation"] = mapping[block["id"]]
    return project


def _jpeg_quality(image):
    tables = getattr(image, "quantization", None)
    if not tables:
        return 95
    table = tables.get(0) if isinstance(tables, dict) else None
    if not table:
        return 95
    table = list(table)
    std = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
           14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
           18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
           49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]
    total = 0.0
    for i, v in enumerate(table):
        if i < len(std) and v > 0:
            total += (v * 100.0) / std[i]
    if not table:
        return 95
    scale = total / len(table)
    if scale <= 100:
        q = (200 - scale) / 2.0
    else:
        q = 5000.0 / scale
    return int(max(1, min(100, round(q))))


def _save_image(canvas, dest, fmt, src_image_path):
    if fmt in ("jpg", "jpeg"):
        quality = 95
        try:
            with Image.open(src_image_path) as original:
                quality = _jpeg_quality(original)
        except Exception:
            pass
        canvas.save(dest, format="JPEG", quality=quality, subsampling=0, optimize=True)
    elif fmt == "webp":
        canvas.save(dest, format="WEBP", quality=95, method=4)
    else:
        canvas.save(dest)


def do_render(project, input_dir, out_dir, cfg):
    from .config import find_cjk_font
    font_path = find_cjk_font(cfg.get("font") or None)
    os.makedirs(out_dir, exist_ok=True)
    output_format = cfg.get("output_format", "png").lower()
    keep_ext = cfg.get("keep_ext", False)
    for page in project["pages"]:
        fmt = output_format
        if keep_ext:
            fmt = os.path.splitext(page["image"])[1].lstrip(".").lower() or "png"
        src = os.path.join(input_dir, page["image"])
        clean = os.path.splitext(src)[0] + ".clean.png"
        if os.path.exists(clean):
            with Image.open(clean) as img:
                canvas = render.render_page(img.convert("RGB"), page, cfg, font_path, erase=False)
        else:
            with Image.open(src) as img:
                canvas = render.render_page(img.convert("RGB"), page, cfg, font_path)
        stem = os.path.splitext(page["image"])[0]
        dest = os.path.join(out_dir, stem + "." + ("jpg" if fmt in ("jpg", "jpeg") else fmt))
        _save_image(canvas, dest, fmt, src)
    return out_dir
