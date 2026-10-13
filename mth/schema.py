SCHEMA_VERSION = 1


def new_project(source_lang, target_lang, chapter=""):
    return {
        "version": SCHEMA_VERSION,
        "meta": {
            "chapter": chapter,
            "source_lang": source_lang,
            "target_lang": target_lang,
        },
        "glossary": {},
        "pages": [],
    }


def new_block(block_id, bbox, ocr="", direction="auto", btype="speech"):
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    return {
        "id": block_id,
        "bbox": [x0, y0, x1, y1],
        "direction": direction,
        "type": btype,
        "ocr": ocr,
        "translation": "",
    }


def new_page(index, image, width, height, blocks=None):
    return {
        "page": index,
        "image": image,
        "width": int(width),
        "height": int(height),
        "blocks": blocks or [],
    }


def iter_blocks(project, need_ocr=False, need_translation=False):
    for page in project["pages"]:
        for block in page["blocks"]:
            if need_ocr and not block.get("ocr"):
                continue
            if need_translation and not block.get("translation"):
                continue
            yield page, block


def block_size(block):
    x0, y0, x1, y1 = block["bbox"]
    return x1 - x0, y1 - y0


def page_of(project, index):
    for page in project["pages"]:
        if page["page"] == index:
            return page
    return None
