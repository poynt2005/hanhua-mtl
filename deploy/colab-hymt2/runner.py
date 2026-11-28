"""Colab L4 one-click translation runner for Hy-MT2-30B-A3B-uncensored-heretic.

Enforces:
- Drive-cached llama.cpp b12019 CUDA server (supports hy_v3, zero Docker needed)
- Local SSD model staging (never mmap Google Drive FUSE)
- 4 hard GPU gates (L4/24GB+ VRAM, CUDA0 device, full layer offload, >=18GB VRAM, Gate >=8 tok/s)
- Faithful Hy-MT2 prompting (no system message, STYLE_ADULT personalization template,
  whole-word case-insensitive glossary in Simplified via tw2sp, up to 3 lines context
  omitted for ui/system, 1 line/request x 4 parallel slots, plain-template fallback)
- Simplified canonical output (t2s) + Taiwan Traditional post-conversion (s2twp default)
- Append-only checkpoint/resume + DONE marker SHA-256 verification
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import schema  # noqa: E402

LLAMA_TAG = "b12019"
PREBUILT_BIN_SPEC = {
    "filename": f"llama-server-l4-{LLAMA_TAG}.tar.gz",
    "size_bytes": 773245209,
    "sha256": "802a169e33dde5e73403c2a36f3d6dbfd2d34d5e9ccd898e79a1eb899bcac947",
    "url": (
        f"https://github.com/poynt2005/hanhua-mtl/releases/download/"
        f"colab-l4-llama-{LLAMA_TAG}/llama-server-l4-{LLAMA_TAG}.tar.gz"
    ),
}

MODEL_SPEC = schema.MODELS["hymt2"]
MODEL_ALIAS = MODEL_SPEC["alias"]
CTX_SIZE = 32768
N_PARALLEL = 4
MIN_TOTAL_VRAM_MIB = 22000.0
MIN_USED_VRAM_MIB = 18000.0
DEFAULT_MIN_TPS = 8.0
TARGET_LANG = "简体中文"

EXIT_OK = 0
EXIT_FATAL = 1
EXIT_PARTIAL = 2
CHECKPOINT_WRITE_RETRIES = 5
CHECKPOINT_RETRY_BASE_DELAY = 0.25
DEFAULT_MAX_UNTRANSLATED_RATIO = 0.001
PROMPT_MAX_LETTER_RUN = 6
RUN_LIMIT_FLOOR = 24
ATTEMPT_RAW_LIMIT = 4000

SAMPLING = {
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 0,
    "repeat_penalty": 1.0,
    "max_tokens": 1024,
}

STYLE_ADULT = (
    "成人向Galgame台词风格：口语化、直白露骨，性器官与性行为使用直接的说法，"
    "不委婉、不省略、不说教；拟声词与喘息译成对应的中文拟声词，"
    "保留原文的拉长音、重复次数与标点（……、～、！）"
)
OUTPUT_ONLY = "只输出译文，不要额外解释，不要输出原文"

GATE_SAMPLES = [
    {"id": "gate-1", "kind": "dialogue", "speaker": None, "text": "Good morning. Did you sleep well?"},
    {"id": "gate-2", "kind": "dialogue", "speaker": None, "text": "Ahh... don't stop, I'm going to cum!"},
    {"id": "gate-3", "kind": "dialogue", "speaker": None, "text": "*squelch* *squelch* Haah... haah..."},
]

_EN_PRESENT_RE = re.compile(r"[A-Za-z]")
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_PREFIX_RE = re.compile(r"^\s*(译文|譯文|翻译|翻譯|Translation)\s*[:：]\s*")
_THINK_RE = re.compile(r"</?think[^>]*>")
_GATE_REFUSAL_RE = re.compile(r"(抱歉|無法|无法|不能提供|I can't|I cannot|I'm sorry|as an AI)", re.I)
_REFUSAL_RE = re.compile(
    r"(as an AI|I cannot fulfill|I can't fulfill|I cannot assist|I cannot translate this|"
    r"I'm sorry,\s*I can(?:'t|not)|不能提供(?:相关|色情|成人|此类)|无法提供(?:相关|色情|成人|此类)|"
    r"无法翻译(?:此类|色情|成人)|违反.*?(?:政策|准则|规范))",
    re.I,
)
_OFFLOAD_RE = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers\s+to\s+GPU", re.I)

# Control-code runs (with trailing spaces) are split points the model never sees.
_SEPARATOR_RE = re.compile(r"((?:(?:" + schema.CONTROL_CODE_RE.pattern + r")[ 　]*)+)")
_LONG_LETTER_RUN_RE = re.compile(r"([A-Za-z])\1{%d,}" % PROMPT_MAX_LETTER_RUN)
_CHAR_RUN_RE = re.compile(r"([一-鿿A-Za-z])\1*")
_SOURCE_RUN_RE = re.compile(r"([A-Za-z])\1*", re.I)
_SOURCE_LOOP_RE = re.compile(r"([A-Za-z][^\n]{0,9}?)\1{4,}", re.I)
_PHRASE_LOOP_RE = re.compile(r"([一-鿿][^\n]{1,7}?)\1{7,}")

_CC_CACHE: dict[str, Any] = {}
_CC_LOCK = threading.Lock()


def _opencc(config: str) -> Any:
    with _CC_LOCK:
        if config not in _CC_CACHE:
            try:
                from opencc import OpenCC
            except ImportError as exc:
                raise RuntimeError(
                    "OpenCC is required: pip install opencc-python-reimplemented"
                ) from exc
            _CC_CACHE[config] = OpenCC(config)
        return _CC_CACHE[config]


def prompt_to_simplified(text: str) -> str:
    """Taiwan Traditional -> Simplified for glossary/speaker inserted into prompt."""
    return _opencc("tw2sp").convert(text) if text else text


def output_to_simplified(text: str) -> str:
    """Normalize any Traditional output to canonical Simplified."""
    return _opencc("t2s").convert(text) if text else text


def to_traditional(text: str, mode: str = "s2twp") -> str:
    if mode not in ("s2twp", "s2t"):
        raise ValueError(f"Unsupported tw_mode {mode!r}; expected 's2twp' or 's2t'")
    return _opencc(mode).convert(text) if text else text


def parse_gpu_offload_log(log_text: str) -> tuple[int, int]:
    matches = _OFFLOAD_RE.findall(log_text or "")
    if not matches:
        raise RuntimeError("GPU offload verification failed: no 'offloaded X/Y layers to GPU' in llama.log")
    offloaded, total = int(matches[-1][0]), int(matches[-1][1])
    if total <= 0 or offloaded != total:
        raise RuntimeError(
            f"GPU offload incomplete: offloaded {offloaded}/{total} layers (CPU fallback forbidden)"
        )
    return offloaded, total


def query_gpu_stats() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15)
    if res.returncode != 0 or not res.stdout.strip():
        raise RuntimeError(f"nvidia-smi failed: {res.stderr.strip() or 'no output'}")
    line = res.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 5:
        raise RuntimeError(f"Unexpected nvidia-smi output: {line!r}")

    def _f(val: str) -> float:
        try:
            return float(val)
        except ValueError:
            return 0.0

    return {
        "name": parts[0],
        "memory_total_mib": _f(parts[1]),
        "memory_used_mib": _f(parts[2]),
        "utilization_gpu": _f(parts[3]),
        "power_draw_w": _f(parts[4]),
    }


def _bin_env(local_bin_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    ld_prev = env.get("LD_LIBRARY_PATH", "")
    parts = [str(local_bin_dir), "/usr/local/cuda/lib64"]
    if ld_prev:
        parts.append(ld_prev)
    env["LD_LIBRARY_PATH"] = ":".join(parts)
    return env


def _clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def _copy_or_link_file(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if src.is_symlink():
        target = os.readlink(src)
        if os.path.basename(target) == target:
            try:
                os.symlink(target, dst)
                return
            except OSError:
                pass
        resolved = src.resolve()
        if resolved.is_file():
            shutil.copyfile(resolved, dst)
        return
    if src.is_file():
        shutil.copyfile(src, dst)


def _safe_extract_tar_gz(tar_path: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        safe_members: list[tarfile.TarInfo] = []
        for member in tf.getmembers():
            parts = PurePosixPath(member.name).parts
            if member.name.startswith("/") or ".." in parts:
                raise RuntimeError(f"Unsafe tar member path rejected: {member.name!r}")
            if member.issym() or member.islnk():
                link_parts = PurePosixPath(member.linkname).parts
                if os.path.isabs(member.linkname) or ".." in link_parts:
                    raise RuntimeError(
                        f"Unsafe tar link target rejected: {member.name!r} -> {member.linkname!r}"
                    )
            safe_members.append(member)
        if "filter" in inspect.signature(tf.extractall).parameters:
            tf.extractall(dst_dir, members=safe_members, filter="tar")
        else:
            tf.extractall(dst_dir, members=safe_members)


def _collect_bin_files(src_root: Path, local_bin_dir: Path) -> bool:
    _clear_dir(local_bin_dir)
    exe_src = src_root / "llama-server"
    if not exe_src.is_file():
        exe_src = next((p for p in src_root.rglob("llama-server") if p.is_file()), None)
    if exe_src is None:
        return False
    local_exe = local_bin_dir / "llama-server"
    shutil.copyfile(exe_src, local_exe)
    local_exe.chmod(0o755)
    seen = {"llama-server"}
    for so_file in sorted(src_root.rglob("*.so*")):
        if (so_file.is_file() or so_file.is_symlink()) and so_file.name not in seen:
            _copy_or_link_file(so_file, local_bin_dir / so_file.name)
            seen.add(so_file.name)
    return True


def _verify_llama_binary(local_exe: Path, local_bin_dir: Path) -> bool:
    if not local_exe.is_file():
        return False
    local_exe.chmod(0o755)
    env = _bin_env(local_bin_dir)
    try:
        chk_ver = subprocess.run(
            [str(local_exe), "--version"],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if chk_ver.returncode != 0:
            print(f"[bin] --version check failed (rc={chk_ver.returncode})")
            return False
        chk_dev = subprocess.run(
            [str(local_exe), "--list-devices"],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        dev_out = f"{chk_dev.stdout}\n{chk_dev.stderr}"
        if chk_dev.returncode != 0 or "CUDA" not in dev_out.upper():
            print(f"[bin] --list-devices has no CUDA device (rc={chk_dev.returncode})")
            return False
        return True
    except Exception as exc:
        print(f"[bin] Binary verification failed: {exc}")
        return False


def _extract_and_verify_bundle(tar_path: Path, local_bin_dir: Path) -> bool:
    extract_tmp = local_bin_dir.parent / "_extract_tmp"
    _clear_dir(extract_tmp)
    try:
        _safe_extract_tar_gz(tar_path, extract_tmp)
        if not _collect_bin_files(extract_tmp, local_bin_dir):
            print(f"[bin] Bundle missing llama-server: {tar_path.name}")
            return False
        return _verify_llama_binary(local_bin_dir / "llama-server", local_bin_dir)
    except Exception as exc:
        print(f"[bin] Extract/verify failed for {tar_path.name}: {exc}")
        return False
    finally:
        shutil.rmtree(extract_tmp, ignore_errors=True)


def _download_url_to_file(
    url: str,
    dst: Path,
    expected_size: int | None = None,
    timeout: int = 180,
    retries: int = 3,
) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(f".{dst.name}.part")
    for attempt in range(1, retries + 1):
        part.unlink(missing_ok=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "hanhua-mtl-colab/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(part, "wb") as fh:
                while True:
                    chunk = resp.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
            if expected_size is not None and part.stat().st_size != expected_size:
                raise RuntimeError(
                    f"size mismatch {part.stat().st_size} != {expected_size}"
                )
            os.replace(part, dst)
            return True
        except urllib.error.HTTPError as exc:
            part.unlink(missing_ok=True)
            if exc.code == 404:
                return False
            print(f"[bin] Download HTTP {exc.code} (attempt {attempt}/{retries}): {url}")
        except Exception as exc:
            part.unlink(missing_ok=True)
            print(f"[bin] Download error (attempt {attempt}/{retries}): {exc}")
        if attempt < retries:
            time.sleep(2 * attempt)
    return False


def ensure_llama_server(drive_root: Path, local_bin_dir: Path) -> Path:
    local_bin_dir.mkdir(parents=True, exist_ok=True)
    local_exe = local_bin_dir / "llama-server"
    spec = PREBUILT_BIN_SPEC
    filename = spec["filename"]
    exp_size = int(spec["size_bytes"])
    exp_sha = spec["sha256"]
    cache_tar = drive_root / "bin" / filename
    cache_sha = drive_root / "bin" / f"{filename}.sha256"

    # Tier 1: Verified prebuilt bundle already in Drive bin/
    if cache_tar.is_file():
        sz = cache_tar.stat().st_size
        if sz == exp_size and schema.sha256_file(cache_tar) == exp_sha:
            print(f"[bin] Found verified Drive prebuilt bundle: {cache_tar.name}")
            if _extract_and_verify_bundle(cache_tar, local_bin_dir):
                print("[bin] Prebuilt llama-server verified OK")
                return local_exe
            raise RuntimeError(
                f"Drive prebuilt bundle {cache_tar.name} failed CUDA binary verification."
            )
        print(f"[bin] Drive bundle size/SHA-256 mismatch ({sz} bytes); re-downloading from GitHub Release...")

    # Tier 2: Download pinned prebuilt bundle from GitHub Release and cache to Drive bin/
    tmp_dir = local_bin_dir.parent / "_prebuilt_tmp"
    _clear_dir(tmp_dir)
    try:
        tmp_tar = tmp_dir / filename
        print(f"[bin] Downloading prebuilt {filename} from GitHub Release...")
        if not _download_url_to_file(spec["url"], tmp_tar, expected_size=exp_size, timeout=300, retries=3):
            raise RuntimeError(f"Failed to download prebuilt llama-server from {spec['url']}")
        got_sha = schema.sha256_file(tmp_tar)
        if got_sha != exp_sha:
            raise RuntimeError(f"Prebuilt llama-server SHA-256 mismatch: {got_sha} != {exp_sha}")
        if not _extract_and_verify_bundle(tmp_tar, local_bin_dir):
            raise RuntimeError("Downloaded prebuilt llama-server failed CUDA binary verification.")
        cache_tar.parent.mkdir(parents=True, exist_ok=True)
        tmp_drive_tar = cache_tar.with_name(f".{cache_tar.name}.tmp")
        try:
            shutil.copyfile(tmp_tar, tmp_drive_tar)
            os.replace(tmp_drive_tar, cache_tar)
            schema.atomic_write_text(cache_sha, f"{exp_sha}  {filename}\n")
        finally:
            tmp_drive_tar.unlink(missing_ok=True)
        print(f"[bin] Cached prebuilt llama-server to {cache_tar}")
        return local_exe
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def stage_model(drive_root: Path, local_models_dir: Path) -> Path:
    drive_model = drive_root / "models" / MODEL_SPEC["filename"]
    if not drive_model.is_file():
        raise FileNotFoundError(
            f"Missing model on Drive: {drive_model}. Run deploy_models_colab.ipynb first."
        )
    actual_size = drive_model.stat().st_size
    if actual_size != MODEL_SPEC["size_bytes"]:
        raise RuntimeError(
            f"Drive model size mismatch: {actual_size} != {MODEL_SPEC['size_bytes']}. "
            "Re-run deploy_models_colab.ipynb."
        )

    local_models_dir.mkdir(parents=True, exist_ok=True)
    local_model = local_models_dir / MODEL_SPEC["filename"]
    if local_model.is_file() and local_model.stat().st_size == MODEL_SPEC["size_bytes"]:
        if schema.sha256_file(local_model) == MODEL_SPEC["sha256"]:
            print(f"[model] Local SSD model already staged & verified: {local_model}")
            return local_model
        local_model.unlink(missing_ok=True)

    tmp_local = local_model.with_name(f".{local_model.name}.tmp")
    print(f"[model] Staging {MODEL_SPEC['size_bytes'] / 1e9:.2f} GB from Drive to local SSD...")
    t0 = time.monotonic()
    shutil.copyfile(drive_model, tmp_local)
    print(f"[model] Copied to SSD in {time.monotonic() - t0:.1f}s; verifying SHA-256 locally...")
    if tmp_local.stat().st_size != MODEL_SPEC["size_bytes"]:
        tmp_local.unlink(missing_ok=True)
        raise RuntimeError("Local staged model size mismatch after copy")
    local_sha = schema.sha256_file(tmp_local)
    if local_sha != MODEL_SPEC["sha256"]:
        tmp_local.unlink(missing_ok=True)
        raise RuntimeError(
            f"Model SHA-256 mismatch: {local_sha} != {MODEL_SPEC['sha256']}. "
            "Re-run deploy_models_colab.ipynb."
        )
    os.replace(tmp_local, local_model)
    print(f"[model] Staged & SHA-256 verified -> {local_model}")
    return local_model


def _http_get_json(url: str, timeout: float = 5.0) -> tuple[int, Any]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", "replace")
        return resp.status, json.loads(raw) if raw else {}


class OutputTruncated(Exception):
    """The server stopped at a token limit, so the text is incomplete."""

    def __init__(self, text: str):
        super().__init__("output truncated (finish_reason=length)")
        self.text = text


def chat_stream(
    api_base: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: float = 180.0,
    retries: int = 3,
    max_tokens: int | None = None,
) -> tuple[str, int, float]:
    """Streaming SSE chat completion using Hy-MT2 model card sampling."""
    payload = dict(SAMPLING, model=model, messages=messages, stream=True)
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    data = json.dumps(payload).encode("utf-8")
    url = api_base.rstrip("/") + "/chat/completions"
    last_err: Exception | None = None

    for attempt in range(retries + 1):
        t0 = time.monotonic()
        pieces: list[str] = []
        chunk_tokens = 0
        usage_tokens: int | None = None
        finish_reason: str | None = None
        try:
            req = urllib.request.Request(
                url,
                data=data,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    try:
                        chunk = json.loads(body)
                    except ValueError:
                        continue
                    usage = chunk.get("usage") or {}
                    if isinstance(usage.get("completion_tokens"), int):
                        usage_tokens = usage["completion_tokens"]
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    if choices[0].get("finish_reason"):
                        finish_reason = choices[0]["finish_reason"]
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        pieces.append(piece)
                        chunk_tokens += 1
            elapsed = max(0.001, time.monotonic() - t0)
            text = "".join(pieces)
            if finish_reason == "length":
                raise OutputTruncated(text)
            if not text:
                raise RuntimeError("stream produced empty content")
            tokens = usage_tokens if usage_tokens and usage_tokens > 0 else max(chunk_tokens, len(text))
            return text, tokens, elapsed
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            if exc.code < 500 and exc.code != 429:
                raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
            last_err = RuntimeError(f"HTTP {exc.code}: {detail}")
        except (urllib.error.URLError, http.client.HTTPException, OSError, RuntimeError) as exc:
            last_err = exc
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"chat_stream failed after {retries + 1} attempts: {last_err}")


def _relevant_terms(text: str, glossary_cn: list[dict[str, str]]) -> list[tuple[str, str]]:
    """Whole-word case-insensitive match for English glossary terms."""
    hits: list[tuple[str, str]] = []
    for ent in glossary_cn:
        src = ent["src"]
        pat = r"(?<![A-Za-z])" + re.escape(src) + r"(?![A-Za-z])"
        if re.search(pat, text, re.IGNORECASE):
            hits.append((src, ent["dst"]))
    return hits


def build_hymt2_messages(
    text: str,
    kind: str,
    glossary_cn: list[dict[str, str]],
    speaker: str | None = None,
    context: list[str] | None = None,
    style: str = STYLE_ADULT,
) -> list[dict[str, str]]:
    """One user message, no system prompt, model card personalization template."""
    reqs: list[str] = []
    if style and kind not in ("ui", "system"):
        reqs.append(style)
    terms = _relevant_terms(text, glossary_cn)
    if terms:
        reqs.append(
            "参考下面的翻译：" + "；".join(f"{src} 翻译成 {dst}" for src, dst in terms)
        )
    if speaker and kind == "dialogue":
        reqs.append(f"说话人是「{prompt_to_simplified(speaker)}」，人名保持一致")
    if context and kind not in ("ui", "system"):
        reqs.append("上文（仅供参考，不要翻译）：" + " / ".join(context[-3:]))
    reqs.append(OUTPUT_ONLY)
    reqs.append(f"将【待翻译文本】翻译为{TARGET_LANG}。")
    body = f"【待翻译文本】\n{text}\n【翻译任务】\n"
    body += "\n".join(f"{i + 1}、{r}" for i, r in enumerate(reqs))
    return [{"role": "user", "content": body}]


def build_hymt2_messages_plain(text: str) -> list[dict[str, str]]:
    """Card default template: fallback when personalization template has mechanical issues."""
    return [
        {
            "role": "user",
            "content": f"将以下文本翻译为{TARGET_LANG}，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}",
        }
    ]


def clean_output(output: str) -> str:
    text = _PREFIX_RE.sub("", output.strip())
    return _THINK_RE.sub("", text).strip()


def compress_long_runs(text: str) -> str:
    """Shorten very long letter runs in the PROMPT only; they push models into repetition loops."""
    return _LONG_LETTER_RUN_RE.sub(lambda m: m.group(1) * PROMPT_MAX_LETTER_RUN, text)


def check_runaway(src_text: str, zh_cn: str) -> list[str]:
    """Reject degenerate repetition and outputs far longer than their English source."""
    src = schema.CONTROL_CODE_RE.sub("", src_text or "")
    out = schema.CONTROL_CODE_RE.sub("", zh_cn or "")
    errs: list[str] = []
    if len(out) > 2 * len(src) + 30:
        errs.append("too_long")
    drawl = max((len(m.group(0)) for m in _SOURCE_RUN_RE.finditer(src)), default=0)
    longest = max((len(m.group(0)) for m in _CHAR_RUN_RE.finditer(out)), default=0)
    if longest > max(RUN_LIMIT_FLOOR, 2 * drawl + 4):
        errs.append("repetition")
    elif not _SOURCE_LOOP_RE.search(src) and any(
        len(set(m.group(1))) > 1 for m in _PHRASE_LOOP_RE.finditer(out)
    ):
        errs.append("repetition")
    return errs


def mechanical_problems(output: str, source: str | None = None) -> list[str]:
    probs: list[str] = []
    text = output.strip()
    if not text:
        probs.append("empty")
        return probs
    if "<think" in text or "</think" in text:
        probs.append("thinking_block")
    if _PREFIX_RE.match(text):
        probs.append("explanatory_prefix")
    if "\n" in text and (source is None or "\n" not in source):
        probs.append("extra_lines")
    if source is not None and schema.extract_placeholders(source) != schema.extract_placeholders(text):
        probs.append("placeholder_mismatch")
    if source is not None:
        if schema.extract_control_codes(source) != schema.extract_control_codes(text):
            probs.append("control_code_mismatch")
        probs.extend(check_runaway(source, text))
    if _REFUSAL_RE.search(text):
        probs.append("refusal")
    letters = sum(c.isascii() and c.isalpha() for c in schema.CONTROL_CODE_RE.sub("", text))
    if not _CJK_RE.search(text) and letters > 3:
        probs.append("not_translated")
    return probs


def start_and_verify_server(
    llama_exe: Path,
    local_bin_dir: Path,
    model_path: Path,
    log_path: Path,
    min_tps: float = DEFAULT_MIN_TPS,
) -> tuple[subprocess.Popen[Any], str]:
    stats = query_gpu_stats()
    print(
        f"[gpu] Detected {stats['name']} | total={stats['memory_total_mib']:.0f} MiB | "
        f"used={stats['memory_used_mib']:.0f} MiB"
    )
    if stats["memory_total_mib"] < MIN_TOTAL_VRAM_MIB:
        raise RuntimeError(
            f"GPU VRAM {stats['memory_total_mib']:.0f} MiB < {MIN_TOTAL_VRAM_MIB:.0f} MiB "
            "(likely T4 instead of L4/A100). Aborting to avoid CPU fallback."
        )

    env = _bin_env(local_bin_dir)
    dev_res = subprocess.run(
        [str(llama_exe), "--list-devices"],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
    )
    if dev_res.returncode != 0 or "CUDA0" not in (dev_res.stdout + dev_res.stderr):
        raise RuntimeError(
            f"Gate 1 failed: CUDA0 not listed by llama-server --list-devices:\n{dev_res.stdout}\n{dev_res.stderr}"
        )
    print("[gpu] Gate 1 PASS: CUDA0 visible to llama-server")

    env.update(
        {
            "LLAMA_ARG_MODEL": str(model_path),
            "LLAMA_ARG_ALIAS": MODEL_ALIAS,
            "LLAMA_ARG_HOST": "127.0.0.1",
            "LLAMA_ARG_PORT": "11434",
            "LLAMA_ARG_N_GPU_LAYERS": "999",
            "LLAMA_ARG_CTX_SIZE": str(CTX_SIZE),
            "LLAMA_ARG_N_PARALLEL": str(N_PARALLEL),
            "LLAMA_ARG_JINJA": "1",
            "LLAMA_ARG_UI": "0",
        }
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", newline="\n") as log_fh:
        proc = subprocess.Popen(
            [
                str(llama_exe),
                "--device", "CUDA0",
                "--cont-batching",
                "-lv", "4",
                "--temp", "0.7",
                "--top-p", "1.0",
                "--top-k", "0",
                "--repeat-penalty", "1.0",
            ],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )

    api_base = "http://127.0.0.1:11434/v1"
    deadline = time.monotonic() + 300.0
    ready = False
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
            raise RuntimeError(f"llama-server exited early (rc={proc.returncode}):\n{tail}")
        try:
            code, health = _http_get_json("http://127.0.0.1:11434/health", timeout=3.0)
            if code == 200 and health.get("status") == "ok":
                mcode, models = _http_get_json(f"{api_base}/models", timeout=3.0)
                ids = [m.get("id") for m in models.get("data", [])]
                if mcode == 200 and MODEL_ALIAS in ids:
                    ready = True
                    break
        except Exception:
            pass
        time.sleep(2.0)

    if not ready:
        proc.terminate()
        raise RuntimeError("Timed out waiting for llama-server /health and /v1/models")

    # Poll briefly because llama.cpp common_log flushes via a background thread
    log_txt = ""
    offloaded, total = 0, 0
    for attempt in range(10):
        log_txt = log_path.read_text(encoding="utf-8", errors="replace")
        try:
            offloaded, total = parse_gpu_offload_log(log_txt)
            break
        except Exception as exc:
            if attempt == 9:
                proc.terminate()
                tail = log_txt[-800:].strip()
                raise RuntimeError(f"{exc}\n--- llama.log tail ---\n{tail}") from exc
            time.sleep(0.5)
    print(f"[gpu] Gate 2 PASS: offloaded {offloaded}/{total} layers to GPU")

    used_stats = query_gpu_stats()
    if used_stats["memory_used_mib"] < MIN_USED_VRAM_MIB:
        proc.terminate()
        raise RuntimeError(
            f"Gate 3 failed: VRAM used {used_stats['memory_used_mib']:.0f} MiB < {MIN_USED_VRAM_MIB:.0f} MiB"
        )
    print(f"[gpu] Gate 3 PASS: VRAM used {used_stats['memory_used_mib']:.0f} MiB")

    run_preflight_gate(api_base, min_tps=min_tps)
    return proc, api_base


def run_preflight_gate(api_base: str, min_tps: float = DEFAULT_MIN_TPS) -> None:
    print("[gate] Running 3-line Hy-MT2 pre-flight gate...")
    total_tokens = 0
    total_sec = 0.0
    for idx, sample in enumerate(GATE_SAMPLES, start=1):
        candidates = (
            build_hymt2_messages(sample["text"], sample["kind"], []),
            build_hymt2_messages_plain(sample["text"]),
        )
        last_probs: list[str] = []
        last_out = ""
        passed = False
        for msgs in candidates:
            out, toks, elapsed = chat_stream(api_base, MODEL_ALIAS, msgs)
            probs = mechanical_problems(out, sample["text"])
            if _GATE_REFUSAL_RE.search(out) and "refusal" not in probs:
                probs.append("refusal")
            if not probs:
                cn = output_to_simplified(clean_output(out))
                total_tokens += toks
                total_sec += elapsed
                print(f"  [gate {idx}/3 OK] {sample['text']} -> {cn}")
                passed = True
                break
            last_probs = probs
            last_out = out
        if not passed:
            raise RuntimeError(f"Gate sample {idx} failed {last_probs}: {last_out!r}")

    tps = total_tokens / max(0.001, total_sec)
    print(f"[gate] Gate 4 PASS: {tps:.1f} tok/s (threshold >= {min_tps:.1f})")
    if tps < min_tps:
        raise RuntimeError(f"Gate 4 failed: throughput {tps:.1f} tok/s < {min_tps:.1f} (CPU fallback?)")


def needs_translation(rec: dict[str, Any]) -> bool:
    if rec.get("override") is not None:
        return False
    return bool(_EN_PRESENT_RE.search(schema.CONTROL_CODE_RE.sub("", rec["text"])))


def make_unit_key(kind: str, speaker: str | None, text: str) -> str:
    """Stable SHA-1 identity independent of sequential display wid."""
    raw = f"{kind}\0{speaker or ''}\0{text}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def validate_unit_translation(src_text: str, zh_cn: str) -> list[str]:
    return mechanical_problems(zh_cn, src_text)


def split_segments(text: str) -> list[tuple[bool, str]]:
    """Split into (translatable, piece) parts; ''.join(pieces) always equals the input."""
    parts: list[tuple[bool, str]] = []
    for idx, piece in enumerate(_SEPARATOR_RE.split(text)):
        if piece:
            parts.append((idx % 2 == 0 and bool(_EN_PRESENT_RE.search(piece)), piece))
    return parts


def assemble_segments(source: str, outputs: list[str]) -> str:
    """Re-insert translated segments between the untouched source separators/literals."""
    it = iter(outputs)
    out: list[str] = []
    for translatable, piece in split_segments(source):
        if not translatable:
            out.append(piece)
            continue
        lead = piece[: len(piece) - len(piece.lstrip(" \u3000"))]
        tail = piece[len(piece.rstrip(" \u3000")):]
        out.append(lead + next(it).strip(" \u3000") + tail)
    return "".join(out)


_ATTEMPT_SINK: Any = None


def _record_attempt(unit: dict[str, Any], segment: str, mode: str, errs: list[str], raw: str) -> None:
    """Keep what the model actually returned for every rejected request."""
    sink = _ATTEMPT_SINK
    if sink is not None:
        sink(
            {
                "unit_keys": [unit["unit_key"]] if unit.get("unit_key") else [],
                "sources": [segment],
                "template": mode,
                "errors": errs,
                "raw": raw[:ATTEMPT_RAW_LIMIT],
                "raw_len": len(raw),
            }
        )


def _prepare_glossary_simplified(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "src": ent["src"],
            "dst": prompt_to_simplified(ent["dst"]),
            "note": ent.get("note", ""),
        }
        for ent in entries
    ]


def _translate_segment(
    api_base: str,
    unit: dict[str, Any],
    segment: str,
    speaker: str | None,
    glossary_cn: list[dict[str, str]],
    context: list[str],
) -> tuple[tuple[str, str | None, list[str]] | None, list[str]]:
    """Three tries: personalization template -> card plain template -> personalization without context."""
    prompt_text = compress_long_runs(segment)
    max_tokens = min(SAMPLING["max_tokens"], max(96, 2 * len(prompt_text) + 48))

    def _rich(ctx: list[str]) -> list[dict[str, str]]:
        return build_hymt2_messages(prompt_text, unit["kind"], glossary_cn, speaker=speaker, context=ctx)

    plan = (
        ("styled", lambda: _rich(context)),
        ("plain", lambda: build_hymt2_messages_plain(prompt_text)),
        ("styled_no_context", lambda: _rich([])),
    )
    last_probs: list[str] = ["unknown"]
    for mode, build in plan:
        try:
            out, _, _ = chat_stream(api_base, MODEL_ALIAS, build(), max_tokens=max_tokens)
        except OutputTruncated as exc:
            last_probs = ["truncated"]
            _record_attempt(unit, segment, mode, last_probs, exc.text)
            continue
        except Exception as exc:
            last_probs = [f"transport:{exc}"]
            _record_attempt(unit, segment, mode, last_probs, "")
            continue
        cleaned = clean_output(out)
        cn = output_to_simplified(cleaned)
        probs = mechanical_problems(out.strip(), segment) or mechanical_problems(cn, segment)
        if probs:
            last_probs = probs
            _record_attempt(unit, segment, mode, probs, out)
            continue
        flags = ["fallback_plain"] if mode == "plain" else []
        if cn != cleaned:
            flags.append("t2s")
        return (cn, cleaned if cn != cleaned else None, flags), []
    return None, last_probs


def _translate_one_unit(
    api_base: str,
    unit: dict[str, Any],
    glossary_cn: list[dict[str, str]],
    context: list[str],
) -> tuple[str | None, str | None, list[str], list[str]]:
    """One request per segment between control codes; the runner re-inserts the codes itself."""
    segments = [piece for translatable, piece in split_segments(unit["text"]) if translatable]
    outputs: list[str] = []
    raws: list[str] = []
    flags: list[str] = []
    seg_context = list(context)
    for idx, segment in enumerate(segments):
        speaker = unit.get("speaker") if idx == 0 else None
        result, probs = _translate_segment(
            api_base, unit, segment.strip(" \u3000"), speaker, glossary_cn, seg_context
        )
        if result is None:
            return None, None, [], probs
        cn, raw_saved, seg_flags = result
        outputs.append(cn)
        raws.append(raw_saved if raw_saved is not None else cn)
        flags.extend(f for f in seg_flags if f not in flags)
        if unit["kind"] not in ("ui", "system"):
            seg_context = (seg_context + [segment.strip(" \u3000")])[-3:]
    joined = assemble_segments(unit["text"], outputs)
    probs = validate_unit_translation(unit["text"], joined)
    if probs:
        _record_attempt(unit, unit["text"], "assembled", probs, joined)
        return None, None, [], probs
    raw_joined = assemble_segments(unit["text"], raws) if raws != outputs else None
    return joined, raw_joined, flags, []


def _load_checkpoint(
    ckpt_path: Path,
    legacy_wid_to_key: dict[str, str] | None = None,
) -> tuple[dict[str, dict[str, Any]], int]:
    done: dict[str, dict[str, Any]] = {}
    migrated = 0
    if not ckpt_path.is_file():
        return done, migrated
    with open(ckpt_path, "r", encoding="utf-8-sig") as handle:
        for raw in handle:
            s = raw.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except ValueError:
                continue
            zh_cn = obj.get("zh_cn")
            if not isinstance(zh_cn, str) or not zh_cn.strip():
                continue
            uk = obj.get("unit_key")
            if isinstance(uk, str) and uk.strip():
                done[uk] = obj
                continue
            wid = obj.get("wid")
            if isinstance(wid, str) and legacy_wid_to_key and wid in legacy_wid_to_key:
                key = legacy_wid_to_key[wid]
                done[key] = {**obj, "unit_key": key}
                migrated += 1
    return done, migrated


def load_overrides_file(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load manual overrides keyed by record id and/or unit_key."""
    by_id: dict[str, str] = {}
    by_key: dict[str, str] = {}
    if not path.is_file():
        return by_id, by_key
    with open(path, "r", encoding="utf-8-sig") as handle:
        for lineno, raw in enumerate(handle, start=1):
            s = raw.strip()
            if not s:
                continue
            obj = json.loads(s)
            val = obj.get("translation")
            if val is None:
                val = obj.get("zh_cn")
            if val is None:
                val = obj.get("override")
            if obj.get("status") == "kept_source" and not str(val or "").strip():
                continue  # review row the operator has not filled in yet
            if not isinstance(val, str) or not val.strip():
                raise schema.SchemaError(
                    f"{path.name}:{lineno}: override requires non-empty translation/zh_cn/override"
                )
            rid = obj.get("id")
            uk = obj.get("unit_key")
            if isinstance(rid, str) and rid.strip():
                by_id[rid] = val
            elif isinstance(uk, str) and uk.strip():
                by_key[uk] = val
            elif isinstance(obj.get("kind"), str) and isinstance(obj.get("text"), str):
                by_key[make_unit_key(obj["kind"], obj.get("speaker"), obj["text"])] = val
            else:
                raise schema.SchemaError(
                    f"{path.name}:{lineno}: override row needs 'id', 'unit_key', or ('kind' + 'text')"
                )
    return by_id, by_key


def _revalidate_checkpoint(
    done_map: dict[str, dict[str, Any]],
    units_by_key: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Drop cached rows that fail the CURRENT validators so they get retranslated."""
    valid: dict[str, dict[str, Any]] = {}
    rejected: dict[str, list[str]] = {}
    for key, row in done_map.items():
        unit = units_by_key.get(key)
        if unit is None:
            valid[key] = row
            continue
        errs = validate_unit_translation(unit["text"], row["zh_cn"])
        if errs:
            rejected[key] = errs
        else:
            valid[key] = row
    return valid, rejected


def _make_attempt_sink(path: Path) -> Any:
    """Append-only diagnostics; never deleted, and a write error never stops the job."""
    lock = threading.Lock()

    def _sink(entry: dict[str, Any]) -> None:
        line = json.dumps({"at": schema.utc_now_iso(), **entry}, ensure_ascii=False) + "\n"
        try:
            with lock, open(path, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(line)
        except OSError as exc:
            print(f"[attempts] warn: could not append {path.name}: {exc}")

    return _sink


def untranslated_limit(total_records: int, max_ratio: float) -> int:
    return int(total_records * max_ratio + 1e-9)


def is_partial_summary(summary: dict[str, Any]) -> bool:
    return bool(
        summary.get("missing", 0)
        or summary.get("no_cjk", 0)
        or summary.get("latin_only", 0)
        or summary.get("over_untranslated_limit", False)
        or summary.get("transport_failed_units", 0)
    )


def load_overrides_file(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load manual overrides keyed by record id and/or unit_key."""
    by_id: dict[str, str] = {}
    by_key: dict[str, str] = {}
    if not path.is_file():
        return by_id, by_key
    with open(path, "r", encoding="utf-8-sig") as handle:
        for lineno, raw in enumerate(handle, start=1):
            s = raw.strip()
            if not s:
                continue
            obj = json.loads(s)
            val = obj.get("translation")
            if val is None:
                val = obj.get("zh_cn")
            if val is None:
                val = obj.get("override")
            if obj.get("status") == "kept_source" and not str(val or "").strip():
                continue  # review row the operator has not filled in yet
            if not isinstance(val, str) or not val.strip():
                raise schema.SchemaError(
                    f"{path.name}:{lineno}: override requires non-empty translation/zh_cn/override"
                )
            rid = obj.get("id")
            uk = obj.get("unit_key")
            if isinstance(rid, str) and rid.strip():
                by_id[rid] = val
            elif isinstance(uk, str) and uk.strip():
                by_key[uk] = val
            elif isinstance(obj.get("kind"), str) and isinstance(obj.get("text"), str):
                by_key[make_unit_key(obj["kind"], obj.get("speaker"), obj["text"])] = val
            else:
                raise schema.SchemaError(
                    f"{path.name}:{lineno}: override row needs 'id', 'unit_key', or ('kind' + 'text')"
                )
    return by_id, by_key


def _revalidate_checkpoint(
    done_map: dict[str, dict[str, Any]],
    units_by_key: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Drop cached rows that fail the CURRENT validators so they get retranslated."""
    valid: dict[str, dict[str, Any]] = {}
    rejected: dict[str, list[str]] = {}
    for key, row in done_map.items():
        unit = units_by_key.get(key)
        if unit is None:
            valid[key] = row
            continue
        errs = validate_unit_translation(unit["text"], row["zh_cn"])
        if errs:
            rejected[key] = errs
        else:
            valid[key] = row
    return valid, rejected


def _make_attempt_sink(path: Path) -> Any:
    """Append-only diagnostics; never deleted, and a write error never stops the job."""
    lock = threading.Lock()

    def _sink(entry: dict[str, Any]) -> None:
        line = json.dumps({"at": schema.utc_now_iso(), **entry}, ensure_ascii=False) + "\n"
        try:
            with lock, open(path, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(line)
        except OSError as exc:
            print(f"[attempts] warn: could not append {path.name}: {exc}")

    return _sink


def untranslated_limit(total_records: int, max_ratio: float) -> int:
    return int(total_records * max_ratio + 1e-9)


def is_partial_summary(summary: dict[str, Any]) -> bool:
    return bool(
        summary.get("missing", 0)
        or summary.get("no_cjk", 0)
        or summary.get("latin_only", 0)
        or summary.get("over_untranslated_limit", False)
        or summary.get("transport_failed_units", 0)
    )


def _append_checkpoint_rows(
    ckpt_path: Path,
    rows: list[dict[str, Any]],
    retries: int = CHECKPOINT_WRITE_RETRIES,
    base_delay: float = CHECKPOINT_RETRY_BASE_DELAY,
) -> None:
    if not rows:
        return
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    last_exc: OSError | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            with open(ckpt_path, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(payload)
                fh.flush()
            return
        except OSError as exc:
            last_exc = exc
            if attempt < retries:
                delay = base_delay * (2 ** (attempt - 1))
                print(
                    f"[ckpt] warn: write failed on {ckpt_path.name} "
                    f"(attempt {attempt}/{retries}): {exc}; retrying in {delay:.2f}s"
                )
                time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _log_gpu_progress(done_count: int, total_count: int, gpu_log: Path, skip_gpu_check: bool) -> None:
    if skip_gpu_check:
        msg = f"[progress] {done_count}/{total_count} unique units"
    else:
        try:
            st = query_gpu_stats()
            msg = (
                f"[progress] {done_count}/{total_count} | GPU: {st['utilization_gpu']:.0f}% | "
                f"VRAM: {st['memory_used_mib']:.0f}/{st['memory_total_mib']:.0f} MiB | "
                f"Power: {st['power_draw_w']:.1f}W"
            )
        except Exception as exc:
            msg = f"[progress] {done_count}/{total_count} | GPU query warn: {exc}"
    print(msg)
    try:
        with open(gpu_log, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(f"{schema.utc_now_iso()} {msg}\n")
    except OSError as exc:
        print(f"[progress] warn: could not append {gpu_log.name}: {exc}")


def _failure_record(unit: dict[str, Any], errs: list[str]) -> dict[str, Any]:
    return {
        "unit_key": unit["unit_key"],
        "wid": unit["wid"],
        "text": unit["text"],
        "errors": errs,
        "expected_placeholders": schema.extract_placeholders(unit["text"]),
        "source_kind": unit["kind"],
    }


def translate_job(
    job: str,
    input_path: Path,
    work_dir: Path,
    output_dir: Path,
    api_base: str,
    glossary_entries: list[dict[str, str]],
    tw_mode: str = "s2twp",
    force: bool = False,
    manifest_wait: float = 60.0,
    skip_gpu_check: bool = False,
    overrides_path: Path | None = None,
    repair_only: bool = False,
    assemble_only: bool = False,
    max_untranslated_ratio: float = DEFAULT_MAX_UNTRANSLATED_RATIO,
) -> dict[str, Any]:
    global _ATTEMPT_SINK
    try:
        if (
            not force
            and not repair_only
            and not assemble_only
            and schema.done_marker_path(output_dir, job).is_file()
        ):
            schema.verify_done_marker(output_dir, job)
            print(f"[job] {job}: DONE marker verified, skipping (use --force to rerun)")
            return {"job": job, "status": "skipped_done"}
    except schema.SchemaError:
        pass

    schema.verify_manifest(input_path, wait_sec=manifest_wait)
    records = schema.validate_jsonl_file(input_path)
    glossary_cn = _prepare_glossary_simplified(glossary_entries)

    eff_overrides = overrides_path if overrides_path is not None else input_path.with_name(f"{job}.overrides.jsonl")
    ov_by_id, ov_by_key = load_overrides_file(eff_overrides) if eff_overrides.is_file() else ({}, {})

    # Legacy checkpoints are keyed by sequential wid. Rebuild that numbering with the OLD
    # "has an ASCII letter" rule (control-code letters included) so rows map to the same text.
    legacy_wid_to_key: dict[str, str] = {}
    legacy_seen: set[str] = set()
    for rec in records:
        if rec.get("override") is not None or not _EN_PRESENT_RE.search(rec["text"]):
            continue
        uk = make_unit_key(rec["kind"], rec.get("speaker"), rec["text"])
        if uk not in legacy_seen:
            legacy_seen.add(uk)
            legacy_wid_to_key[f"w{len(legacy_seen):06d}"] = uk

    work_units: list[dict[str, Any]] = []
    seen_keys: dict[str, dict[str, Any]] = {}
    rec_to_unit_key: dict[str, str] = {}
    override_cn: dict[str, str] = {}
    passthrough_cn: dict[str, str] = {}

    for rec in records:
        rid = rec["id"]
        if rec.get("override") is not None:
            override_cn[rid] = output_to_simplified(str(rec["override"]))
            continue
        if rid in ov_by_id:
            override_cn[rid] = output_to_simplified(ov_by_id[rid])
            continue
        uk = make_unit_key(rec["kind"], rec.get("speaker"), rec["text"])
        if uk in ov_by_key:
            override_cn[rid] = output_to_simplified(ov_by_key[uk])
            continue
        if not needs_translation(rec):
            passthrough_cn[rid] = rec["text"]
            continue
        if uk not in seen_keys:
            unit_obj = {
                "unit_key": uk,
                "wid": f"w{len(work_units) + 1:06d}",
                "file": rec["file"],
                "kind": rec["kind"],
                "speaker": rec.get("speaker"),
                "text": rec["text"],
            }
            seen_keys[uk] = unit_obj
            work_units.append(unit_obj)
        rec_to_unit_key[rid] = uk

    ckpt_path = work_dir / f"{job}.checkpoint.jsonl"
    failures_path = work_dir / f"{job}.failures.jsonl"
    gpu_log = work_dir / f"{job}.gpu.log"
    done_map, migrated = _load_checkpoint(ckpt_path, legacy_wid_to_key=legacy_wid_to_key)
    if migrated:
        print(f"[ckpt] Migrated {migrated} legacy wid-only checkpoint rows by unit_key")
    done_map, rejected = _revalidate_checkpoint(done_map, seen_keys)
    if rejected:
        reasons: dict[str, int] = {}
        for errs in rejected.values():
            for err in errs:
                reasons[err] = reasons.get(err, 0) + 1
        print(f"[ckpt] {len(rejected)} cached units failed current validation, not reused: {reasons}")

    todo_units = [u for u in work_units if u["unit_key"] not in done_map]
    cached_count = len(work_units) - len(todo_units)
    print(
        f"[job] {job}: total={len(records)} | passthrough={len(passthrough_cn)} | "
        f"overrides={len(override_cn)} | unique_work={len(work_units)} | "
        f"cached={cached_count} | todo={len(todo_units)}"
    )

    by_file: dict[str, list[dict[str, Any]]] = {}
    for u in todo_units:
        by_file.setdefault(u["file"], []).append(u)
    queues: list[list[dict[str, Any]]] = list(by_file.values())
    if len(queues) == 1 and len(queues[0]) > 16:
        items = queues[0]
        step = max(4, (len(items) + N_PARALLEL - 1) // N_PARALLEL)
        queues = [items[i : i + step] for i in range(0, len(items), step)]

    if assemble_only:
        queues = []
    # This run rewrites the outputs: drop any earlier DONE marker until the new result passes.
    schema.done_marker_path(output_dir, job).unlink(missing_ok=True)

    ckpt_lock = threading.Lock()
    gpu_log_lock = threading.Lock()
    progress_counter = {"count": cached_count}
    all_failures: list[dict[str, Any]] = []

    def _flush_pending(pending: list[dict[str, Any]]) -> None:
        if not pending:
            return
        should_log_gpu = False
        cur_done = 0
        with ckpt_lock:
            _append_checkpoint_rows(ckpt_path, pending)
            for row in pending:
                done_map[row["unit_key"]] = row
            prev = progress_counter["count"]
            progress_counter["count"] += len(pending)
            cur_done = progress_counter["count"]
            should_log_gpu = (cur_done // 50) > (prev // 50)
            pending.clear()
        if should_log_gpu:
            with gpu_log_lock:
                _log_gpu_progress(cur_done, len(work_units), gpu_log, skip_gpu_check)

    def _run_queue(q_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
        q_failures: list[dict[str, Any]] = []
        context: list[str] = []
        pending_rows: list[dict[str, Any]] = []
        try:
            for u in q_units:
                use_ctx = [] if u["kind"] in ("ui", "system") else context
                cn, raw_saved, flags, probs = _translate_one_unit(api_base, u, glossary_cn, use_ctx)
                if cn is not None:
                    if u["kind"] not in ("ui", "system"):
                        context = (context + [u["text"]])[-3:]
                    row: dict[str, Any] = {"unit_key": u["unit_key"], "wid": u["wid"], "zh_cn": cn}
                    if raw_saved is not None:
                        row["raw"] = raw_saved
                    if flags:
                        row["flags"] = flags
                    pending_rows.append(row)
                    if len(pending_rows) >= 10:
                        _flush_pending(pending_rows)
                else:
                    q_failures.append(_failure_record(u, probs))
        finally:
            _flush_pending(pending_rows)
        return q_failures

    if queues:
        _ATTEMPT_SINK = _make_attempt_sink(work_dir / f"{job}.attempts.jsonl")
        try:
            with ThreadPoolExecutor(max_workers=N_PARALLEL) as pool:
                for q_fail in pool.map(_run_queue, queues):
                    all_failures.extend(q_fail)
        finally:
            _ATTEMPT_SINK = None

    if assemble_only:
        previous: dict[str, dict[str, Any]] = {}
        if failures_path.is_file():
            with open(failures_path, "r", encoding="utf-8-sig") as handle:
                for raw in handle:
                    if raw.strip():
                        obj = json.loads(raw)
                        if obj.get("unit_key"):
                            previous[obj["unit_key"]] = obj
        for unit in todo_units:
            all_failures.append(
                previous.get(unit["unit_key"])
                or _failure_record(unit, rejected.get(unit["unit_key"]) or ["not_translated"])
            )

    if all_failures:
        failures_txt = "".join(json.dumps(f, ensure_ascii=False) + "\n" for f in all_failures)
        schema.atomic_write_text(failures_path, failures_txt)
    elif failures_path.exists():
        failures_path.unlink()
    failure_by_key = {f["unit_key"]: f for f in all_failures}

    final_done, _ = _load_checkpoint(ckpt_path, legacy_wid_to_key=legacy_wid_to_key)
    final_done, _ = _revalidate_checkpoint(final_done, seen_keys)
    review: dict[str, dict[str, Any]] = {}
    kept_rows = 0
    missing_ids: list[str] = []
    cn_rows: list[dict[str, Any]] = []
    tw_rows: list[dict[str, Any]] = []
    zh_cn_texts: list[str] = []
    zh_tw_texts: list[str] = []

    for line_no, rec in enumerate(records, start=1):
        rid = rec["id"]
        kept_source = False
        if rid in override_cn:
            cn_val = override_cn[rid]
        elif rid in passthrough_cn:
            cn_val = passthrough_cn[rid]
        else:
            uk = rec_to_unit_key[rid]
            entry = final_done.get(uk)
            if entry:
                cn_val = entry["zh_cn"]
            elif uk in failure_by_key:
                # Last resort: ship the untouched source line and list it for review.
                kept_source = True
                kept_rows += 1
                cn_val = rec["text"]
                if uk not in review:
                    review[uk] = {
                        "unit_key": uk,
                        "status": "kept_source",
                        "translation": "",
                        "text": rec["text"],
                        "kind": rec["kind"],
                        "speaker": rec.get("speaker"),
                        "first_line": line_no,
                        "first_id": rid,
                        "rows": 0,
                        "errors": failure_by_key[uk]["errors"],
                    }
                review[uk]["rows"] += 1
            else:
                missing_ids.append(rid)
                cn_val = ""
        if kept_source:
            tw_val = cn_val  # untouched source, never run through OpenCC
        else:
            tw_val = to_traditional(cn_val, mode=tw_mode) if cn_val else ""
        if cn_val and not kept_source:
            zh_cn_texts.append(cn_val)
            zh_tw_texts.append(tw_val)
        base_out = {
            "id": rid,
            "file": rec["file"],
            "kind": rec["kind"],
            "speaker": rec.get("speaker"),
            "text": rec["text"],
        }
        cn_rows.append({**base_out, "translation": cn_val})
        tw_rows.append({**base_out, "translation": tw_val})

    limit = untranslated_limit(len(records), max_untranslated_ratio)
    transport_units = sum(
        1 for item in review.values() if any(e.startswith("transport:") for e in item["errors"])
    )
    qa_summary = {
        "job": job,
        "total_records": len(records),
        "unique_work_units": len(work_units),
        "passthrough_records": len(passthrough_cn),
        "manual_overrides": len(override_cn),
        "missing": len(missing_ids),
        "failures": len(review),
        "kept_source_units": len(review),
        "kept_source_rows": kept_rows,
        "kept_source_limit_rows": limit,
        "over_untranslated_limit": kept_rows > limit,
        "transport_failed_units": transport_units,
        "cache_rejected_units": len(rejected),
        "tw_mode": tw_mode,
        "watch_chars": schema.count_watch_chars(zh_cn_texts, zh_tw_texts),
        "generated_at_utc": schema.utc_now_iso(),
    }

    cn_out = output_dir / f"{job}.zh_CN.jsonl"
    tw_out = output_dir / f"{job}.zh_TW.jsonl"
    qa_out = output_dir / f"{job}.qa_report.json"
    review_out = output_dir / f"{job}.review.jsonl"

    schema.atomic_write_text(
        cn_out, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cn_rows)
    )
    schema.atomic_write_text(
        tw_out, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in tw_rows)
    )
    schema.atomic_write_json(qa_out, qa_summary)
    schema.atomic_write_text(
        review_out,
        "".join(
            json.dumps(item, ensure_ascii=False) + "\n"
            for item in sorted(review.values(), key=lambda i: i["first_line"])
        ),
    )

    complete = not is_partial_summary(qa_summary)
    if complete:
        marker = schema.build_done_marker(job, [cn_out, tw_out, qa_out, review_out], qa_summary)
        schema.atomic_write_json(schema.done_marker_path(output_dir, job), marker)
    model_rows = len(records) - len(missing_ids) - kept_rows - len(override_cn) - len(passthrough_cn)
    status = "DONE" if complete else "PARTIAL"
    print(f"[{status}] {job}: {len(records) - len(missing_ids)} / {len(records)} 列已輸出 -> {tw_out.name}")
    print(f"  模型翻譯通過檢查：{model_rows}")
    print(f"  人工指定／免翻：  {len(override_cn)} / {len(passthrough_cn)}")
    print(f"  待你檢查（保留英文原文）：{kept_rows}（上限 {limit}）")
    if review:
        print(f"  檢查清單：{review_out.name}")
    if transport_units:
        print(f"  [warn] {transport_units} 個單元沒有拿到模型回應（連線／server 問題），請用 --repair-only 重跑")
    if missing_ids:
        print(f"  [warn] missing={len(missing_ids)}")
    return qa_summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Colab L4 Hy-MT2-heretic translation runner")
    ap.add_argument("--drive-root", default=f"/content/drive/MyDrive/{schema.DRIVE_ROOT_NAME}")
    ap.add_argument("--api-base", help="Use an already-running OpenAI-compatible /v1 endpoint")
    ap.add_argument("--tw-mode", default="s2twp", choices=("s2twp", "s2t"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--repair-only", action="store_true", help="Resume from checkpoint and repair remaining units only")
    ap.add_argument("--job", help="Run only the named job (<job>.jsonl)")
    ap.add_argument("--overrides", help="Optional path to overrides JSONL file")
    ap.add_argument(
        "--assemble-only",
        action="store_true",
        help="No server, no model calls: rebuild outputs from checkpoint + overrides",
    )
    ap.add_argument(
        "--max-untranslated-ratio",
        type=float,
        default=DEFAULT_MAX_UNTRANSLATED_RATIO,
        help="Share of rows that may keep the source text and still count as DONE",
    )
    ap.add_argument("--skip-gpu-check", action="store_true")
    ap.add_argument("--manifest-wait", type=float, default=60.0)
    ap.add_argument("--min-tps", type=float, default=DEFAULT_MIN_TPS)
    args = ap.parse_args(argv)

    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")

    dirs = schema.ensure_drive_tree(args.drive_root)
    input_dir = dirs["hymt2_input"]
    work_dir = dirs["hymt2_work"]
    output_dir = dirs["hymt2_output"]

    for p in sorted(input_dir.iterdir()):
        if not p.is_file():
            continue
        if p.name.startswith(".") or p.name.lower() == "desktop.ini":
            continue
        if p.name.endswith(".manifest.json") or p.name.endswith(".overrides.jsonl"):
            continue
        if p.suffix.lower() != ".jsonl":
            raise schema.SchemaError(
                f"Unexpected non-.jsonl file in {input_dir}: {p.name}. Remove it before running."
            )

    job_files = sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() == ".jsonl"
        and not p.name.endswith(".overrides.jsonl")
    )
    if args.job:
        target_name = schema.validate_job_name(args.job)
        job_files = [p for p in job_files if p.stem == target_name]
        if not job_files:
            raise FileNotFoundError(f"Requested --job {target_name!r} not found in {input_dir}")
    if not job_files:
        print(f"[runner] No .jsonl jobs found in {input_dir}")
        return EXIT_OK

    glossary_path = dirs["root"] / schema.SUBDIR / "glossary.json"
    glossary_entries = schema.validate_glossary_file(glossary_path) if glossary_path.is_file() else []
    for jf in job_files:
        schema.verify_manifest(jf, wait_sec=args.manifest_wait)
        schema.validate_jsonl_file(jf)
    print(f"[runner] Pre-validation passed for {len(job_files)} job(s)")

    proc: subprocess.Popen[Any] | None = None
    try:
        if args.assemble_only:
            api_base = ""
        elif args.api_base:
            api_base = args.api_base.rstrip("/")
            if not args.skip_gpu_check:
                run_preflight_gate(api_base, min_tps=args.min_tps)
        else:
            if not args.skip_gpu_check:
                stats0 = query_gpu_stats()
                print(
                    f"[gpu] Gate 0: {stats0['name']} | total={stats0['memory_total_mib']:.0f} MiB"
                )
                if stats0["memory_total_mib"] < MIN_TOTAL_VRAM_MIB:
                    raise RuntimeError(
                        f"Gate 0 failed: GPU VRAM {stats0['memory_total_mib']:.0f} MiB < "
                        f"{MIN_TOTAL_VRAM_MIB:.0f} MiB (likely T4/CPU instead of L4/A100)."
                    )
            local_bin = Path("/content/bin")
            local_models = Path("/content/models")
            llama_log = Path("/content/llama.log")
            llama_exe = ensure_llama_server(dirs["root"], local_bin)
            model_path = stage_model(dirs["root"], local_models)
            proc, api_base = start_and_verify_server(
                llama_exe, local_bin, model_path, llama_log, min_tps=args.min_tps
            )

        partial_jobs: list[dict[str, Any]] = []
        for jf in job_files:
            summary = translate_job(
                job=jf.stem,
                input_path=jf,
                work_dir=work_dir,
                output_dir=output_dir,
                api_base=api_base,
                glossary_entries=glossary_entries,
                tw_mode=args.tw_mode,
                force=args.force,
                manifest_wait=args.manifest_wait,
                skip_gpu_check=args.skip_gpu_check,
                overrides_path=Path(args.overrides) if args.overrides else None,
                repair_only=args.repair_only,
                assemble_only=args.assemble_only,
                max_untranslated_ratio=args.max_untranslated_ratio,
            )
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited unexpectedly during job {jf.stem!r} (rc={proc.returncode}); "
                    "completed progress was saved to checkpoint.jsonl"
                )
            if is_partial_summary(summary):
                partial_jobs.append(summary)
        if partial_jobs:
            print(
                "[runner] PARTIAL_COMPLETE: all jobs finished running and outputs are saved; "
                ".DONE.json NOT written for:"
            )
            for pj in partial_jobs:
                print(
                    f"  - {pj['job']}: 待補 {pj.get('kept_source_rows', 0)} 列"
                    f"（上限 {pj.get('kept_source_limit_rows', 0)}） missing={pj.get('missing', 0)} "
                    f"-> {output_dir / (pj['job'] + '.review.jsonl')}"
                )
            return EXIT_PARTIAL
        return EXIT_OK
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        local_log = Path("/content/llama.log")
        if local_log.is_file():
            try:
                shutil.copyfile(local_log, work_dir / "llama.log")
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
