import base64
import http.client
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request


class ProviderError(RuntimeError):
    pass


def _data_uri(path):
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as handle:
        raw = handle.read()
    return "data:%s;base64,%s" % (mime, base64.b64encode(raw).decode("ascii"))


def _extract_json(text):
    if not text:
        raise ValueError("empty response")
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    chunk = text[start:idx + 1]
                    try:
                        return json.loads(chunk)
                    except json.JSONDecodeError:
                        break
    raise ValueError("no JSON found in response: %s" % text[:300])


class _Retryable(RuntimeError):
    """A transient transport failure worth retrying (connection dropped, reset)."""


class OpenAICompat:
    def __init__(self, base, key, timeout=180, stream=True, retries=3):
        if not key:
            raise ProviderError("missing API key (config api_key / MTH_API_KEY)")
        self.base = base.rstrip("/")
        self.key = key
        self.timeout = timeout
        self.stream = stream
        self.retries = max(1, int(retries))

    def _request(self, payload, stream):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.base + "/chat/completions", data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", "Bearer " + self.key)
        if stream:
            req.add_header("Accept", "text/event-stream")
        return req

    def _plain(self, payload):
        req = self._request(dict(payload, stream=False), False)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", "replace")[:500]
            raise ProviderError("HTTP %s: %s" % (err.code, detail))
        except (urllib.error.URLError, OSError, http.client.HTTPException) as err:
            raise _Retryable("network error: %s" % err)
        return body["choices"][0]["message"]["content"]

    def _stream(self, payload):
        """Streaming keeps bytes flowing while the model generates.

        A non-streaming request is silent for the whole generation, and the
        host's port forward drops an idle connection after ~15s (observed:
        ~1 in 5 requests failed with RemoteDisconnected / WinError 10054).
        """
        req = self._request(dict(payload, stream=True), True)
        pieces = []
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    piece = (choices[0].get("delta") or {}).get("content")
                    if piece:
                        pieces.append(piece)
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", "replace")[:500]
            raise ProviderError("HTTP %s: %s" % (err.code, detail))
        except (urllib.error.URLError, OSError, http.client.HTTPException) as err:
            raise _Retryable("stream dropped: %s" % err)
        text = "".join(pieces)
        if not text:
            raise _Retryable("stream produced no content")
        return text

    def chat(self, model, messages, temperature=0.3, max_tokens=4096, json_mode=False, top_p=None):
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        attempts = []
        if self.stream:
            attempts.append(self._stream)
        attempts.append(self._plain)

        last = None
        for attempt_fn in attempts:
            for attempt in range(self.retries):
                try:
                    return attempt_fn(payload)
                except _Retryable as err:
                    last = err
                    if attempt + 1 < self.retries:
                        time.sleep(1.5 * (attempt + 1))
        raise ProviderError("request failed after retries: %s" % last)

    def vision(self, model, prompt, image_path, system=None, temperature=0.0):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _data_uri(image_path)}},
            ],
        })
        return self.chat(model, messages, temperature=temperature, max_tokens=8192)


def build_client(cfg, mock=False):
    if mock:
        return Mock()
    return OpenAICompat(
        cfg["api_base"],
        cfg.get("api_key", ""),
        timeout=int(cfg.get("timeout", 180)),
        stream=bool(cfg.get("stream", True)),
        retries=int(cfg.get("retries", 3)),
    )


class Mock:
    def __init__(self):
        pass

    def chat(self, model, messages, **kwargs):
        user = ""
        for message in messages:
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                user = message["content"]
        marker = "将下面的文本从日文翻译成简体中文："
        if marker in user or user.startswith("将下面的日文文本翻译成中文"):
            body = user.split(marker, 1)[1] if marker in user else user.split("：", 1)[1]
            lines = body.lstrip("\n").split("\n")
            return "\n".join("〔機翻〕" + line for line in lines)
        payload = None
        try:
            payload = json.loads(user)
        except (ValueError, TypeError):
            payload = None
        if isinstance(payload, dict) and "候選術語" in payload:
            out = {}
            for item in payload["候選術語"]:
                src = item.get("原文")
                if src:
                    out[src] = {"zh": "〔" + src + "〕", "type": "name"}
            return json.dumps(out, ensure_ascii=False)
        if isinstance(payload, dict) and "待修正" in payload:
            glossary = payload.get("術語表", {})
            out = {}
            for item in payload.get("待修正", []):
                text = item.get("現譯", "")
                ocr = item.get("原文", "")
                for src, target in glossary.items():
                    if src in ocr:
                        text = text.replace(src, target)
                        if item.get("必須包含") and item["必須包含"] not in text:
                            text = item["必須包含"] + " " + text
                out[item.get("id")] = text
            return json.dumps(out, ensure_ascii=False)
        source = {}
        if isinstance(payload, dict):
            items = payload.get("原文") or payload.get("source") or []
            for item in items:
                source[str(item.get("id"))] = item.get("ja") or item.get("text") or ""
        out = {key: ("〔機翻〕" + (val or "")) for key, val in source.items()}
        return json.dumps(out, ensure_ascii=False)

    def vision(self, model, prompt, image_path, system=None, temperature=0.0):
        from PIL import Image
        with Image.open(image_path) as img:
            w, h = img.size
        box = [int(w * 0.12), int(h * 0.08), int(w * 0.88), int(h * 0.22)]
        return json.dumps([{
            "bbox": box,
            "text": "テスト用のテキストです",
            "direction": "vertical",
            "type": "speech",
        }], ensure_ascii=False)


VISION_SYSTEM = (
    "You are an OCR engine for manga. You detect every text region and read it. "
    "Always answer with a single JSON array and nothing else."
)

VISION_PROMPT = (
    "This is a manga page of size {w}x{h} pixels. "
    "Find every text region (speech bubbles, captions, sound effects). "
    "Return a JSON array. Each element: "
    '{{"bbox": [x0, y0, x1, y1], "text": "<original text>", '
    '"direction": "vertical"|"horizontal", "type": "speech"|"narration"|"sfx"}}. '
    "bbox uses pixel coordinates with origin at top-left. "
    "Reading order: right to left, top to bottom. Output JSON only."
)


def vision_extract(client, model, image_path, width, height):
    prompt = VISION_PROMPT.format(w=width, h=height)
    content = client.vision(model, prompt, image_path, system=VISION_SYSTEM)
    data = _extract_json(content)
    if isinstance(data, dict):
        for key in ("blocks", "regions", "texts", "items", "data"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ProviderError("vision extract did not return a list")
    blocks = []
    for item in data:
        bbox = item.get("bbox") or item.get("box") or item.get("rect")
        if not bbox or len(bbox) != 4:
            continue
        blocks.append({
            "bbox": [int(round(float(v))) for v in bbox],
            "ocr": (item.get("text") or item.get("ocr") or "").strip(),
            "direction": item.get("direction", "auto"),
            "type": item.get("type", "speech"),
        })
    return blocks


TRANSLATE_SYSTEM = (
    "你是專業的漫畫翻譯，負責日文到繁體中文（台灣用語）的翻譯。"
    "你會收到一個 JSON 陣列，元素包含 id 與原文。"
    "只輸出一個 JSON 物件，鍵為 id（字串），值為譯文（繁體中文）。"
    "保留角色語氣、口語、狀聲詞；不要加解釋；不要合併或省略任何 id。"
    "若有提供術語表，請嚴格遵守其譯名。"
)


def _translate_openai(client, model, pairs, glossary=None, target_lang="zh-TW", system=None):
    glossary = glossary or {}
    sys_text = (system or TRANSLATE_SYSTEM) + "目標語言：" + target_lang + "。"
    user = {"原文": pairs}
    if glossary:
        user["術語表"] = glossary
    messages = [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]
    content = client.chat(model, messages, temperature=0.2, max_tokens=8192, json_mode=True)
    try:
        data = _extract_json(content)
    except ValueError:
        data = None
    result = {}
    if isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, str):
                result[str(key)] = val
            elif isinstance(val, dict):
                result[str(key)] = val.get("translation") or val.get("zh") or val.get("text") or ""
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "id" in item:
                result[str(item["id"])] = item.get("translation") or item.get("zh") or item.get("text") or ""
    if not result:
        raise ProviderError("translation response had no usable entries: %s" % str(content)[:300])
    return result


SAKURA_SYSTEM = (
    "你是一个视觉小说翻译模型，可以通顺地使用给定的术语表以指定的风格将日文翻译成简体中文，"
    "并联系上下文正确使用人称代词，注意不要混淆使役态和被动态的主语和宾语，"
    "不要擅自添加原文中没有的特殊符号，也不要擅自增加或减少换行。"
)


def _sakura_messages(lines, system=None, glossary=None, history=None):
    sys_text = system or SAKURA_SYSTEM
    parts = []
    if history:
        parts.append("历史翻译：" + "\n".join(history))
    gl = ""
    if glossary:
        gl = "\n".join("%s->%s" % (k, v) for k, v in glossary.items())
    parts.append("参考以下术语表（可为空，格式为src->dst #备注）：\n" + gl)
    parts.append("根据以上术语表的对应关系和备注，结合历史剧情和上下文，"
                 "将下面的文本从日文翻译成简体中文：\n" + "\n".join(lines))
    user = "\n\n".join(parts)
    return [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": user},
    ]


def _split_lines(content, expected):
    text = content.strip()
    fenced = re.search(r"```(?:\w+)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    lines = [line.strip() for line in text.split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    if len(lines) == expected + 1 and re.match(r"^[\s【\[]*[Ss]core[:：]?\s*[\d.]+", lines[-1]):
        lines.pop()
    if len(lines) == expected + 1 and re.match(r"^\d+(\.\d+)?$", lines[-1]):
        lines.pop()
    return lines


def _translate_sakura(client, model, pairs, glossary=None, system=None, history=None):
    lines = [p.get("ja") or "" for p in pairs]
    content = client.chat(model, _sakura_messages(lines, system, glossary, history),
                          temperature=0.3, top_p=0.8, max_tokens=4096)
    out_lines = _split_lines(content, len(lines))
    if len(out_lines) != len(lines):
        result = {}
        for pair in pairs:
            single = client.chat(model, _sakura_messages([pair.get("ja") or ""], system, glossary),
                                 temperature=0.3, top_p=0.8, max_tokens=1024)
            got = _split_lines(single, 1)
            result[str(pair["id"])] = got[0] if got else ""
        return result
    return {str(pair["id"]): out_lines[i] for i, pair in enumerate(pairs)}


def translate_batch(client, model, pairs, glossary=None, target_lang="zh-TW",
                    profile="openai", system=None, history=None):
    if profile == "sakura":
        return _translate_sakura(client, model, pairs, glossary, system, history)
    return _translate_openai(client, model, pairs, glossary, target_lang, system)


def ping(client, cfg, profile=None):
    profile = profile or cfg.get("profile", "openai")
    if profile == "sakura":
        reply = client.chat(cfg["text_model"], _sakura_messages(["おはよう。"]),
                            temperature=0.1, max_tokens=64)
    else:
        reply = client.chat(cfg["text_model"],
                            [{"role": "user", "content": "reply with the single word: ok"}],
                            temperature=0, max_tokens=16)
    return reply.strip()
