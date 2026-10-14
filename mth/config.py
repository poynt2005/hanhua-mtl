import json
import os

DEFAULTS = {
    "profile": "openai",
    "api_base": "https://api.deepseek.com/v1",
    "api_key": "",
    "text_model": "deepseek-chat",
    "vision_model": "gpt-4o-mini",
    "source_lang": "ja",
    "target_lang": "zh-TW",
    "system_prompt": "",
    "glossary_file": "",
    "ocr_url": "",
    "ocr_api_key": "",
    "font": "",
    "stroke": 0,
    "padding": 3,
    "inpaint": "fill",
    "translate_chunk": 3000,
    "translate_max_lines": 40,
    "keep_sfx": True,
    "convert_tw": True,
    "timeout": 300,
    # Streaming keeps the socket busy during generation. A non-streaming request
    # is silent for the whole generation and the GPU host's port forward drops an
    # idle connection after ~15s (observed: ~1 in 5 requests failed with
    # RemoteDisconnected / WinError 10054).
    "stream": True,
    "retries": 3,
    "pattern": "*.png",
}

_ENV = {
    "api_base": "MTH_API_BASE",
    "api_key": "MTH_API_KEY",
    "text_model": "MTH_TEXT_MODEL",
    "vision_model": "MTH_VISION_MODEL",
    "source_lang": "MTH_SOURCE_LANG",
    "target_lang": "MTH_TARGET_LANG",
    "font": "MTH_FONT",
}


def load_config(path=None, overrides=None):
    cfg = dict(DEFAULTS)
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    for key, env in _ENV.items():
        val = os.environ.get(env)
        if val:
            cfg[key] = val
    if overrides:
        for key, val in overrides.items():
            if val is not None:
                cfg[key] = val
    return cfg


def find_cjk_font(explicit=None):
    candidates = [
        explicit,
        os.environ.get("MTH_FONT"),
        r"C:\Windows\Fonts\msjh.ttc",
        r"C:\Windows\Fonts\msjhbd.ttc",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\mingliu.ttc",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\kaiu.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(
        "No CJK font found. Set config.font or env MTH_FONT to a .ttf/.ttc path."
    )
