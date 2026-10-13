import json
import re
from collections import Counter

from . import providers

KATA = re.compile(r"[\u30A1-\u30FA\u30FC]{2,}")
KANJI = re.compile(r"[\u4E00-\u9FFF\u3005]{2,}")
LATIN = re.compile(r"[A-Za-z][A-Za-z0-9]{2,}")


def extract_candidates(project, min_count=2, top_k=300):
    text = "\n".join(b.get("ocr", "") for p in project["pages"] for b in p["blocks"])
    counter = Counter()
    for rx in (KATA, KANJI, LATIN):
        for match in rx.findall(text):
            counter[match] += 1
    items = [(term, count) for term, count in counter.items() if count >= min_count and len(term) >= 2]
    items.sort(key=lambda pair: (-pair[1], -len(pair[0])))
    return [{"src": term, "count": count} for term, count in items[:top_k]]


def contexts_for(project, candidates, max_per_term=2):
    wanted = {c["src"] for c in candidates}
    found = {src: [] for src in wanted}
    for page in project["pages"]:
        for block in page["blocks"]:
            text = block.get("ocr", "")
            for src in wanted:
                if src in text and len(found[src]) < max_per_term and text not in found[src]:
                    found[src].append(text)
    return found


PROPOSE_SYSTEM = (
    "你是漫畫漢化的術語校對員。使用者會給你一份「候選術語」清單（含出現次數與上下文）。"
    "請只保留**專有名詞**：人名、招式、組織、地名、專有術語、狀聲詞專名。"
    "丟棄一般詞彙。對每個保留的術語給出**繁體中文（台灣用語）**的標準譯名，同一作品譯名必須唯一。"
    "只輸出 JSON 物件：{\"原文\": {\"zh\": \"標準譯名\", \"type\": \"name|term|sfx\", \"note\": \"\"}}。"
)


def propose_glossary(client, model, candidates, contexts, target_lang, temperature=0.2):
    payload = {
        "目標語言": target_lang,
        "候選術語": [
            {"原文": c["src"], "出現次數": c["count"], "上下文": contexts.get(c["src"], [])}
            for c in candidates
        ],
    }
    content = client.chat(
        model,
        [{"role": "system", "content": PROPOSE_SYSTEM},
         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        temperature=temperature, max_tokens=8192, json_mode=True,
    )
    try:
        data = providers._extract_json(content)
    except ValueError:
        data = {}
    terms = []
    if isinstance(data, dict):
        for src, val in data.items():
            if isinstance(val, dict):
                zh = (val.get("zh") or "").strip()
                btype = val.get("type", "term")
            else:
                zh = str(val).strip()
                btype = "term"
            if not zh or btype == "common":
                continue
            count = next((c["count"] for c in candidates if c["src"] == src), 0)
            terms.append({"src": src, "zh": zh, "type": btype, "count": count, "status": "proposed"})
    terms.sort(key=lambda t: -t["count"])
    return terms


def write_glossary(path, terms, target_lang):
    payload = {"version": 1, "target_lang": target_lang, "terms": terms}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def load_glossary(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    terms = data.get("terms") if isinstance(data, dict) else None
    if terms is None and isinstance(data, dict):
        flat = {k: v for k, v in data.items() if isinstance(v, str)}
        terms = [{"src": k, "zh": v, "type": "term", "count": 0, "status": "approved"} for k, v in flat.items()]
    terms = terms or []
    flat_map = {t["src"]: t["zh"] for t in terms if t.get("src") and t.get("zh")}
    return terms, flat_map


def apply_to_project(project, flat_map):
    project.setdefault("glossary", {})
    project["glossary"].update(flat_map)
    return project


ENFORCE_SYSTEM = (
    "你是翻譯校對員。使用者給你術語表與一批「待修正譯文」。"
    "請把每條譯文中的專有名詞改成術語表指定的標準譯名，其餘文字保持不變，不要增刪內容。"
    "只輸出 JSON 物件：{\"id\": \"修正後譯文\"}。"
)


def _affected(project, flat_map):
    hits = []
    for page in project["pages"]:
        for block in page["blocks"]:
            ocr = block.get("ocr") or ""
            translation = block.get("translation") or ""
            if not ocr or not translation:
                continue
            for src, target in flat_map.items():
                if src in ocr and target and target not in translation:
                    hits.append({"id": block["id"], "ocr": ocr, "translation": translation,
                                 "src": src, "target": target})
                    break
    return hits


def enforce_glossary(project, flat_map, client, cfg, batch=40):
    hits = _affected(project, flat_map)
    if not hits:
        return 0
    model = cfg["text_model"]
    fixed = 0
    for i in range(0, len(hits), batch):
        chunk = hits[i:i + batch]
        subset = {}
        for item in chunk:
            subset[item["src"]] = item["target"]
        payload = {
            "術語表": subset,
            "待修正": [{"id": it["id"], "原文": it["ocr"],
                        "現譯": it["translation"], "必須包含": it["target"]} for it in chunk],
        }
        content = client.chat(
            model,
            [{"role": "system", "content": ENFORCE_SYSTEM},
             {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            temperature=0.0, max_tokens=8192, json_mode=True,
        )
        try:
            data = providers._extract_json(content)
        except ValueError:
            data = {}
        if not isinstance(data, dict):
            continue
        for item in chunk:
            new_text = data.get(item["id"]) or data.get(item["id"].replace("-", ""))
            if isinstance(new_text, str) and new_text.strip():
                _set_translation(project, item["id"], new_text.strip())
                fixed += 1
    return fixed


def _set_translation(project, block_id, text):
    for page in project["pages"]:
        for block in page["blocks"]:
            if block["id"] == block_id:
                block["translation"] = text
                return


def verify_glossary(project, flat_map):
    violations = []
    variants = {}
    for page in project["pages"]:
        for block in page["blocks"]:
            ocr = block.get("ocr") or ""
            translation = block.get("translation") or ""
            for src, target in flat_map.items():
                if src and src in ocr:
                    bucket = variants.setdefault(src, Counter())
                    bucket[target if (target and target in translation) else "<<不一致>>"] += 1
                    if target and target not in translation:
                        violations.append({"id": block["id"], "src": src,
                                           "expected": target, "translation": translation})
    return violations, variants
