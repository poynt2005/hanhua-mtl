import glob
import os
import re
import shutil

from PIL import Image

from . import ocr, providers, render
from .schema import new_block, new_page, new_project


# Every page format an archive can hold. Kept in sync with archive.IMAGE_EXT: a
# page this list misses is extracted but never rendered, and repack then aborts
# with "translated page missing" (or, worse, silently drops the page).
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")


def list_images(input_dir, pattern):
    """All page images in a folder, whatever their extension.

    `pattern` is only a hint for the dominant format: real archives mix
    extensions (a .webp cover next to .jpg pages), so selecting a single glob
    loses pages. Never narrow the result to one extension here.
    """
    files = set(glob.glob(os.path.join(input_dir, pattern)))
    for ext in IMAGE_EXTS:
        files.update(glob.glob(os.path.join(input_dir, "*" + ext)))
        files.update(glob.glob(os.path.join(input_dir, "*" + ext.upper())))
    return sorted(files)


def reading_order(blocks):
    if not blocks:
        return blocks
    heights = [b["bbox"][3] - b["bbox"][1] for b in blocks]
    row = max(20, int(sum(heights) / len(heights) * 0.6))
    return sorted(blocks, key=lambda b: (round(b["bbox"][1] / row), -b["bbox"][0]))


_TEXT_MARK = re.compile(r"[_\-]t(\d+)", re.I)
_CLEAN_MARK = re.compile(r"[_\-]n(\d+)", re.I)


def pair_by_names(project, mapping):
    """Point pages that carry text at their textless twin, decided by name.

    CG sets ship the same drawing twice, and the reliable evidence is the
    filename, not the pixels: `..._t00001` is the printed copy of `..._n00001`.
    Pixel similarity was measured and rejected - inside a single-scene CG set
    every clean page looks like every printed page (true pair MAD p50 18.3 vs
    best false candidate 19.2), so an image-based rule would happily typeset one
    page's dialogue onto another page's drawing.

    Pairing therefore requires an explicit `t<n>` / `n<n>` marker with the same
    number, plus identical pixel size. Anything else keeps the normal
    erase path.
    """
    entries = {item["local"]: item["entry"] for item in mapping}
    texted, clean = [], []
    for page in project["pages"]:
        entry = entries.get(page["image"], "")
        if _page_has_text(page):
            texted.append((page, entry))
        elif _CLEAN_MARK.search(entry):
            clean.append((page, entry))

    by_number = {}
    for page, entry in clean:
        by_number.setdefault(_CLEAN_MARK.search(entry).group(1), []).append(page)

    used = set()
    paired = 0
    for page, entry in texted:
        marker = _TEXT_MARK.search(entry)
        if not marker:
            continue
        for twin in by_number.get(marker.group(1), []):
            if twin["page"] in used:
                continue
            if (twin["width"], twin["height"]) != (page["width"], page["height"]):
                continue
            page["clean_source"] = twin["image"]
            # The twin is a textless copy by name, so whatever the detector
            # thought it saw on it is noise: leave that page untouched.
            twin["blocks"] = []
            used.add(twin["page"])
            paired += 1
            break
    return paired


MAX_BLOCK_AREA = 0.20


def _drop_implausible_blocks(page):
    """Remove boxes that cannot be a speech bubble.

    On clean CG art the detector occasionally fires on texture and produces a
    box covering a quarter to most of the page; manga-ocr then hallucinates a
    short line inside it. Typesetting that fills the box with page-sized
    characters - it destroyed half of one archive before this filter existed.
    Real bubbles are a small fraction of the page, so anything above
    MAX_BLOCK_AREA is dropped rather than trusted.
    """
    total = float(page["width"] * page["height"]) or 1.0
    keep = []
    for block in page["blocks"]:
        x0, y0, x1, y1 = block["bbox"]
        if (x1 - x0) * (y1 - y0) / total > MAX_BLOCK_AREA:
            continue
        keep.append(block)
    page["blocks"] = keep


def _overlap_ratio(first, second):
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = float((ix1 - ix0) * (iy1 - iy0))
    smaller = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
    return inter / smaller if smaller else 0.0


def merge_overlapping_blocks(page, threshold=0.55):
    """Merge boxes that describe the same run of text.

    On art with text printed straight onto it, the detector often returns two
    boxes for one vertical column: a tight one and a loose one that contains it.
    Translating both and drawing both puts two different sentences on top of
    each other - the delivered page becomes an unreadable smear. Merging them
    keeps one block per column, with the whole column's text in reading order.
    """
    blocks = page["blocks"]
    if len(blocks) < 2:
        return 0
    parent = list(range(len(blocks)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    merged = 0
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            if _overlap_ratio(blocks[i]["bbox"], blocks[j]["bbox"]) > threshold:
                root_i, root_j = find(i), find(j)
                if root_i != root_j:
                    parent[root_j] = root_i
                    merged += 1
    if not merged:
        return 0

    groups = {}
    for i, block in enumerate(blocks):
        groups.setdefault(find(i), []).append(block)

    rebuilt = []
    for members in groups.values():
        if len(members) == 1:
            rebuilt.append(members[0])
            continue
        ordered = reading_order(members)
        xs0 = min(b["bbox"][0] for b in members)
        ys0 = min(b["bbox"][1] for b in members)
        xs1 = max(b["bbox"][2] for b in members)
        ys1 = max(b["bbox"][3] for b in members)
        head = ordered[0]
        rebuilt.append({
            "id": head["id"],
            "bbox": [xs0, ys0, xs1, ys1],
            "direction": head.get("direction", "auto"),
            "type": head.get("type", "speech"),
            "ocr": "".join((b.get("ocr") or "") for b in ordered),
            "translation": "".join((b.get("translation") or "") for b in ordered),
        })
    rebuilt.sort(key=lambda b: (b["bbox"][1], -b["bbox"][0]))
    page["blocks"] = rebuilt
    return merged


def _clear_listed_pages(project, spec):
    """Blank pages listed explicitly as textless copies in the config.

    Automatic detection of clean copies was tried and rejected: on a CG set the
    detector's OCR volume on a clean page can sit close enough to a sparse real
    page that any threshold either sprays text on art or leaves Japanese behind.
    An explicit range per archive is unambiguous and costs nothing to apply.
    """
    cleared = 0
    for item in spec or []:
        try:
            first, last = int(item[0]), int(item[1])
        except (TypeError, IndexError, ValueError):
            continue
        for page in project["pages"]:
            if first <= page["page"] <= last and page["blocks"]:
                page["blocks"] = []
                cleared += 1
    return cleared


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
    for page in project["pages"]:
        _drop_implausible_blocks(page)
        merge_overlapping_blocks(page)
    cleared = _clear_listed_pages(project, cfg.get("skip_pages"))
    if cleared:
        print("extract: %d page(s) are listed as textless copies; left untouched"
              % cleared)
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


def _page_has_text(page):
    """True if OCR found any text on this page.

    Pages with no detected text (clean duplicates are common in CG sets) carry
    nothing to erase or typeset, so they must be passed through untouched: a
    decode/encode round trip would only add JPEG generation loss.
    """
    for block in page["blocks"]:
        if (block.get("ocr") or "").strip():
            return True
    return False


def do_render(project, input_dir, out_dir, cfg):
    from .config import find_cjk_font
    font_path = find_cjk_font(cfg.get("font") or None)
    os.makedirs(out_dir, exist_ok=True)
    output_format = cfg.get("output_format", "png").lower()
    keep_ext = cfg.get("keep_ext", False)
    passed_through = 0
    paired = 0
    for page in project["pages"]:
        fmt = output_format
        if keep_ext:
            fmt = os.path.splitext(page["image"])[1].lstrip(".").lower() or "png"
        normalised = "jpg" if fmt in ("jpg", "jpeg") else fmt
        src = os.path.join(input_dir, page["image"])
        clean = os.path.splitext(src)[0] + ".clean.png"
        src_fmt = os.path.splitext(src)[1].lstrip(".").lower()
        if (normalised == ("jpg" if src_fmt in ("jpg", "jpeg") else src_fmt)
                and not _page_has_text(page)):
            # A page with no text is delivered as the original file: the clean
            # copy (if the server made one) is the detector's guess at what to
            # erase, and on a textless page that guess is exactly the damage we
            # are trying to avoid.
            stem = os.path.splitext(page["image"])[0]
            shutil.copyfile(src, os.path.join(out_dir, stem + "." + normalised))
            passed_through += 1
            continue
        clean_src = page.get("clean_source")
        if clean_src:
            # Typeset onto the textless twin: no erase, no artifacts. The twin is
            # only ever set when the detector found no text on it and the drawing
            # matches, so this cannot swap one page's art for another's.
            with Image.open(os.path.join(input_dir, clean_src)) as img:
                canvas = render.render_page(img.convert("RGB"), page, cfg, font_path,
                                            erase=False)
            paired += 1
        elif os.path.exists(clean):
            with Image.open(clean) as img:
                canvas = render.render_page(img.convert("RGB"), page, cfg, font_path, erase=False)
        else:
            with Image.open(src) as img:
                canvas = render.render_page(img.convert("RGB"), page, cfg, font_path)
        stem = os.path.splitext(page["image"])[0]
        dest = os.path.join(out_dir, stem + "." + ("jpg" if fmt in ("jpg", "jpeg") else fmt))
        _save_image(canvas, dest, fmt, src)
    if passed_through:
        print("render: %d page(s) had no text and were passed through untouched"
              % passed_through)
    if paired:
        print("render: %d page(s) were typeset onto their textless twin (no inpainting)"
              % paired)
    return out_dir
