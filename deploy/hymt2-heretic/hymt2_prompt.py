"""Reference prompt + client for the hymt2-heretic image (EN -> zh-TW, adult galgames).

Stdlib only. Copy the prompt builder into the galgame client as its `hymt2`
profile, or import this file. Also a CLI:

    python hymt2_prompt.py --api-base http://<host>:<port>/v1 gate
    python hymt2_prompt.py --api-base http://<host>:<port>/v1 tr "Don't stop..."
    python hymt2_prompt.py --api-base ... tr --glossary glossary.json --speaker Yui "..."

Design (why it looks like this):
  * Hy-MT2 has NO system prompt. Everything goes in one user message.
  * Several requirements at once (style + glossary + context) use the model
    card's "personalization" template - the only official template that carries
    more than one instruction. Its wording is kept verbatim; only the numbered
    requirements are ours.
  * ONE script line per request. The server has 4 slots and only 3B active
    parameters, so per-line requests are fast, and line alignment can never
    break (no merged/split lines to repair).
  * Streaming, with retries: a non-streaming request is silent while the model
    generates and a rented host's port forward drops idle sockets after ~15 s
    (see deploy/VASTAI_PLAYBOOK.md).
  * Sampling = the model card's recommendation for 30B-A3B. Send it explicitly:
    a client's values override the server defaults, so a client still sending
    Sakura's 0.3 / 0.8 would silently change behaviour.
  * The model translates into SIMPLIFIED Chinese - its strongest Chinese (most
    of its Chinese data, and most of the slang/SFX vocabulary, is Simplified).
    Everything the model sees is kept in one script: glossary targets and
    speaker names are converted to Simplified before prompting. The client then
    converts the result to Taiwan Traditional with OpenCC `s2twp` (characters
    AND Taiwan vocabulary; context-aware for one-to-many characters such as
    干 -> 幹/乾, 发 -> 發/髮). Needs `pip install opencc-python-reimplemented`.
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

MODEL = "hy-mt2-heretic"
# The model's strongest target; the card requires the full Chinese name.
# Traditional is produced afterwards by OpenCC (to_traditional), never by the model.
TARGET_LANG = "简体中文"

_CC = {}


def _opencc(config):
    """OpenCC converter, created once. Raises a clear error if not installed."""
    if config not in _CC:
        try:
            from opencc import OpenCC
        except ImportError as exc:
            raise RuntimeError("OpenCC is required: pip install opencc-python-reimplemented") from exc
        _CC[config] = OpenCC(config)
    return _CC[config]


def to_simplified(text):
    """Taiwan Traditional -> Simplified, for anything we put INTO the prompt."""
    return _opencc("tw2sp").convert(text) if text else text


def to_traditional(text):
    """Model output (Simplified) -> Taiwan Traditional with Taiwan vocabulary."""
    return _opencc("s2twp").convert(text) if text else text

SAMPLING = {"temperature": 0.7, "top_p": 1.0, "top_k": 0,
            "repeat_penalty": 1.0, "max_tokens": 1024}

# The house style for adult scenes. Decensoring removed refusals; this line is
# what stops the model from euphemising. Edit to taste; keep it one sentence.
STYLE_ADULT = ("成人向Galgame台词风格：口语化、直白露骨，性器官与性行为使用直接的说法，"
               "不委婉、不省略、不说教；拟声词与喘息译成对应的中文拟声词，"
               "保留原文的拉长音、重复次数与标点（……、～、！）")

OUTPUT_ONLY = "只输出译文，不要额外解释，不要输出原文"


def _relevant_terms(text, glossary):
    """Only the glossary entries that occur in this line (case-insensitive)."""
    hits = []
    for src, dst in (glossary or {}).items():
        pat = r"(?<![A-Za-z])" + re.escape(src) + r"(?![A-Za-z])"
        if re.search(pat, text, re.IGNORECASE):
            hits.append((src, dst))
    return hits


def build_messages(text, glossary=None, speaker=None, context=None, style=STYLE_ADULT):
    """One user message, personalization template (model card, Chinese version)."""
    reqs = []
    if style:
        reqs.append(style)
    terms = _relevant_terms(text, glossary)
    if terms:
        # glossaries are usually kept in Traditional; the model sees Simplified only
        reqs.append("参考下面的翻译：" + "；".join(
            "%s 翻译成 %s" % (src, to_simplified(dst)) for src, dst in terms))
    if speaker:
        reqs.append("说话人是「%s」，人名保持一致" % to_simplified(speaker))
    if context:
        reqs.append("上文（仅供参考，不要翻译）：" + " / ".join(context[-3:]))
    reqs.append(OUTPUT_ONLY)
    reqs.append("将【待翻译文本】翻译为%s。" % TARGET_LANG)
    body = "【待翻译文本】\n%s\n【翻译任务】\n" % text
    body += "\n".join("%d、%s" % (i + 1, r) for i, r in enumerate(reqs))
    return [{"role": "user", "content": body}]


def build_messages_plain(text):
    """The card's default template - the fallback when the rich one misbehaves."""
    return [{"role": "user", "content":
             "将以下文本翻译为%s，注意只需要输出翻译后的结果，不要额外解释：\n\n%s"
             % (TARGET_LANG, text)}]


def chat(api_base, messages, timeout=120, retries=3):
    """Streaming chat completion; returns the concatenated content."""
    payload = dict(SAMPLING, model=MODEL, messages=messages, stream=True)
    data = json.dumps(payload).encode("utf-8")
    url = api_base.rstrip("/") + "/chat/completions"
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=data, method="POST",
                                         headers={"Content-Type": "application/json"})
            out = []
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    delta = json.loads(chunk)["choices"][0].get("delta", {})
                    out.append(delta.get("content") or "")
            return "".join(out)
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            last = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code < 500:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("chat failed after %d retries: %s" % (retries, last))


_CJK = re.compile(r"[㐀-鿿]")
_PREFIX = re.compile(r"^\s*(译文|譯文|翻译|翻譯|Translation)\s*[:：]\s*")


def problems(output, source=None):
    """Mechanical checks on one translated line. Empty list = looks fine."""
    p = []
    text = output.strip()
    if not text:
        p.append("empty")
    if "<think" in text or "</think" in text:
        p.append("thinking block")
    if _PREFIX.match(text):
        p.append("explanatory prefix")
    if "\n" in text and (source is None or "\n" not in source):
        p.append("extra lines")
    letters = sum(c.isascii() and c.isalpha() for c in text)
    if text and not _CJK.search(text) and letters > 3:
        p.append("not translated")
    return p


def clean(output):
    text = _PREFIX.sub("", output.strip())
    return re.sub(r"</?think[^>]*>", "", text).strip()


def translate(api_base, text, glossary=None, speaker=None, context=None, traditional=True):
    """Rich prompt first; on a mechanical problem retry once with the plain one.

    Returns Taiwan Traditional (OpenCC s2twp) by default; traditional=False
    returns the model's Simplified output untouched.
    """
    out = chat(api_base, build_messages(text, glossary, speaker, context))
    if problems(out, text):
        out = chat(api_base, build_messages_plain(text))
    out = clean(out)
    return to_traditional(out) if traditional else out


# Gate: one plain line, one explicit line, one SFX line. Each must come back as
# a single non-empty Chinese line with no refusal, no explanation, no thinking.
GATE = [
    "Good morning. Did you sleep well?",
    "Ahh... don't stop, I'm going to cum!",
    "*squelch* *squelch* Haah... haah...",
]
_REFUSAL = re.compile(r"(抱歉|無法|无法|不能提供|I can't|I cannot|I'm sorry|as an AI)", re.I)


def gate(api_base):
    ok = True
    for line in GATE:
        out = chat(api_base, build_messages(line))
        bad = problems(out, line) + (["refusal"] if _REFUSAL.search(out) else [])
        ok &= not bad
        print("%-5s %s -> %s => %s %s" % ("OK" if not bad else "FAIL", line, out.strip(),
                                          to_traditional(clean(out)), bad if bad else ""))
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--api-base", required=True, help="e.g. http://host:11434/v1")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gate", help="3-line pre-flight check; exit 1 on any failure")
    t = sub.add_parser("tr", help="translate one line")
    t.add_argument("text")
    t.add_argument("--glossary", help="JSON file: {\"english term\": \"中文\"}")
    t.add_argument("--speaker")
    t.add_argument("--show-prompt", action="store_true")
    t.add_argument("--simplified", action="store_true", help="print the raw model output")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # Chinese output on a Windows console
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    if a.cmd == "gate":
        return 0 if gate(a.api_base) else 1
    gl = None
    if a.glossary:
        with open(a.glossary, encoding="utf-8-sig") as f:
            gl = json.load(f)
    if a.show_prompt:
        print(build_messages(a.text, gl, a.speaker)[0]["content"], file=sys.stderr)
    print(translate(a.api_base, a.text, gl, a.speaker, traditional=not a.simplified))
    return 0


if __name__ == "__main__":
    sys.exit(main())
