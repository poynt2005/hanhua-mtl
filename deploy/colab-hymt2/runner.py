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


def chat_stream(
    api_base: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: float = 180.0,
    retries: int = 3,
) -> tuple[str, int, float]:
    """Streaming SSE chat completion using Hy-MT2 model card sampling."""
    payload = dict(SAMPLING, model=model, messages=messages, stream=True)
    data = json.dumps(payload).encode("utf-8")
    url = api_base.rstrip("/") + "/chat/completions"
    last_err: Exception | None = None

    for attempt in range(retries + 1):
        t0 = time.monotonic()
        pieces: list[str] = []
        chunk_tokens = 0
        usage_tokens: int | None = None
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
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        pieces.append(piece)
                        chunk_tokens += 1
            elapsed = max(0.001, time.monotonic() - t0)
            text = "".join(pieces)
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
    if _REFUSAL_RE.search(text):
        probs.append("refusal")
    letters = sum(c.isascii() and c.isalpha() for c in text)
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
    return bool(_EN_PRESENT_RE.search(rec["text"]))


def _prepare_glossary_simplified(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "src": ent["src"],
            "dst": prompt_to_simplified(ent["dst"]),
            "note": ent.get("note", ""),
        }
        for ent in entries
    ]


def _translate_one_unit(
    api_base: str,
    unit: dict[str, Any],
    glossary_cn: list[dict[str, str]],
    context: list[str],
) -> tuple[str | None, str | None, list[str], list[str]]:
    """Rich personalization template first; fallback once with card plain template."""
    msgs = build_hymt2_messages(
        unit["text"],
        unit["kind"],
        glossary_cn,
        speaker=unit.get("speaker"),
        context=context,
    )
    last_probs: list[str] = ["unknown"]
    try:
        out, _, _ = chat_stream(api_base, MODEL_ALIAS, msgs)
        probs = mechanical_problems(out, unit["text"])
        if not probs:
            cleaned = clean_output(out)
            cn = output_to_simplified(cleaned)
            flags = ["t2s"] if cn != cleaned else []
            return cn, (cleaned if flags else None), flags, []
        last_probs = probs
    except Exception as exc:
        last_probs = [f"transport:{exc}"]

    # Fallback once with the official plain template
    try:
        plain_out, _, _ = chat_stream(
            api_base, MODEL_ALIAS, build_hymt2_messages_plain(unit["text"])
        )
        probs2 = mechanical_problems(plain_out, unit["text"])
        if not probs2:
            cleaned = clean_output(plain_out)
            cn = output_to_simplified(cleaned)
            flags = ["fallback_plain"] + (["t2s"] if cn != cleaned else [])
            return cn, (cleaned if cn != cleaned else None), flags, []
        last_probs = probs2
    except Exception as exc:
        last_probs = [f"fallback_transport:{exc}"]

    return None, None, [], last_probs


def _load_checkpoint(ckpt_path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not ckpt_path.is_file():
        return done
    with open(ckpt_path, "r", encoding="utf-8-sig") as handle:
        for raw in handle:
            s = raw.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except ValueError:
                continue
            wid = obj.get("wid")
            zh_cn = obj.get("zh_cn")
            if isinstance(wid, str) and isinstance(zh_cn, str) and zh_cn.strip():
                done[wid] = obj
    return done


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
    with open(gpu_log, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{schema.utc_now_iso()} {msg}\n")


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
) -> dict[str, Any]:
    try:
        if not force and schema.done_marker_path(output_dir, job).is_file():
            schema.verify_done_marker(output_dir, job)
            print(f"[job] {job}: DONE marker verified, skipping (use --force to rerun)")
            return {"job": job, "status": "skipped_done"}
    except schema.SchemaError:
        pass

    schema.verify_manifest(input_path, wait_sec=manifest_wait)
    records = schema.validate_jsonl_file(input_path)
    glossary_cn = _prepare_glossary_simplified(glossary_entries)

    work_units: list[dict[str, Any]] = []
    key_to_wid: dict[tuple[str, str, str], str] = {}
    rec_to_wid: dict[str, str] = {}
    passthrough_cn: dict[str, str] = {}

    for rec in records:
        if rec.get("override") is not None:
            passthrough_cn[rec["id"]] = output_to_simplified(str(rec["override"]))
            continue
        if not needs_translation(rec):
            passthrough_cn[rec["id"]] = rec["text"]
            continue
        key = (rec["kind"], rec.get("speaker") or "", rec["text"])
        if key not in key_to_wid:
            wid = f"w{len(work_units) + 1:06d}"
            key_to_wid[key] = wid
            work_units.append(
                {
                    "wid": wid,
                    "file": rec["file"],
                    "kind": rec["kind"],
                    "speaker": rec.get("speaker"),
                    "text": rec["text"],
                }
            )
        rec_to_wid[rec["id"]] = key_to_wid[key]

    ckpt_path = work_dir / f"{job}.checkpoint.jsonl"
    failures_path = work_dir / f"{job}.failures.jsonl"
    gpu_log = work_dir / f"{job}.gpu.log"
    done_map = _load_checkpoint(ckpt_path)

    todo_units = [u for u in work_units if u["wid"] not in done_map]
    print(
        f"[job] {job}: total={len(records)} | passthrough={len(passthrough_cn)} | "
        f"unique_work={len(work_units)} | cached={len(done_map)} | todo={len(todo_units)}"
    )

    by_file: dict[str, list[dict[str, Any]]] = {}
    for u in todo_units:
        by_file.setdefault(u["file"], []).append(u)
    queues: list[list[dict[str, Any]]] = list(by_file.values())
    if len(queues) == 1 and len(queues[0]) > 16:
        items = queues[0]
        step = max(4, (len(items) + N_PARALLEL - 1) // N_PARALLEL)
        queues = [items[i : i + step] for i in range(0, len(items), step)]

    ckpt_lock = threading.Lock()
    gpu_log_lock = threading.Lock()
    progress_counter = {"count": len(done_map)}
    all_failures: list[dict[str, Any]] = []

    def _flush_pending(pending: list[dict[str, Any]]) -> None:
        if not pending:
            return
        should_log_gpu = False
        cur_done = 0
        with ckpt_lock:
            with open(ckpt_path, "a", encoding="utf-8", newline="\n") as fh:
                for row in pending:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    done_map[row["wid"]] = row
                fh.flush()
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
                    row: dict[str, Any] = {"wid": u["wid"], "zh_cn": cn}
                    if raw_saved is not None:
                        row["raw"] = raw_saved
                    if flags:
                        row["flags"] = flags
                    pending_rows.append(row)
                    if len(pending_rows) >= 10:
                        _flush_pending(pending_rows)
                else:
                    q_failures.append({"wid": u["wid"], "text": u["text"], "errors": probs})
        finally:
            _flush_pending(pending_rows)
        return q_failures

    if queues:
        with ThreadPoolExecutor(max_workers=N_PARALLEL) as pool:
            for q_fail in pool.map(_run_queue, queues):
                all_failures.extend(q_fail)

    if all_failures:
        failures_txt = "".join(json.dumps(f, ensure_ascii=False) + "\n" for f in all_failures)
        schema.atomic_write_text(failures_path, failures_txt)
        print(f"[job] {job}: {len(all_failures)} units failed -> {failures_path}")
    elif failures_path.exists():
        failures_path.unlink()

    final_done = _load_checkpoint(ckpt_path)
    missing_ids: list[str] = []
    cn_rows: list[dict[str, Any]] = []
    tw_rows: list[dict[str, Any]] = []
    zh_cn_texts: list[str] = []
    zh_tw_texts: list[str] = []

    for rec in records:
        rid = rec["id"]
        if rid in passthrough_cn:
            cn_val = passthrough_cn[rid]
        else:
            wid = rec_to_wid[rid]
            entry = final_done.get(wid)
            if not entry:
                missing_ids.append(rid)
                cn_val = ""
            else:
                cn_val = entry["zh_cn"]
        tw_val = to_traditional(cn_val, mode=tw_mode) if cn_val else ""
        if cn_val:
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

    qa_summary = {
        "job": job,
        "total_records": len(records),
        "unique_work_units": len(work_units),
        "passthrough_records": len(passthrough_cn),
        "missing": len(missing_ids),
        "failures": len(all_failures),
        "tw_mode": tw_mode,
        "watch_chars": schema.count_watch_chars(zh_cn_texts, zh_tw_texts),
        "generated_at_utc": schema.utc_now_iso(),
    }

    cn_out = output_dir / f"{job}.zh_CN.jsonl"
    tw_out = output_dir / f"{job}.zh_TW.jsonl"
    qa_out = output_dir / f"{job}.qa_report.json"

    schema.atomic_write_text(
        cn_out, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cn_rows)
    )
    schema.atomic_write_text(
        tw_out, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in tw_rows)
    )
    schema.atomic_write_json(qa_out, qa_summary)

    if not missing_ids and not all_failures:
        marker = schema.build_done_marker(job, [cn_out, tw_out, qa_out], qa_summary)
        schema.atomic_write_json(schema.done_marker_path(output_dir, job), marker)
        print(f"[job] {job}: COMPLETE 100% ({len(records)}/{len(records)}) -> {tw_out.name}")
    else:
        print(
            f"[job] {job}: INCOMPLETE missing={len(missing_ids)} failures={len(all_failures)}; "
            "DONE marker NOT written"
        )
    return qa_summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Colab L4 Hy-MT2-heretic translation runner")
    ap.add_argument("--drive-root", default=f"/content/drive/MyDrive/{schema.DRIVE_ROOT_NAME}")
    ap.add_argument("--api-base", help="Use an already-running OpenAI-compatible /v1 endpoint")
    ap.add_argument("--tw-mode", default="s2twp", choices=("s2twp", "s2t"))
    ap.add_argument("--force", action="store_true")
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
        if p.name.endswith(".manifest.json"):
            continue
        if p.suffix.lower() != ".jsonl":
            raise schema.SchemaError(
                f"Unexpected non-.jsonl file in {input_dir}: {p.name}. Remove it before running."
            )

    job_files = sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() == ".jsonl"
    )
    if not job_files:
        print(f"[runner] No .jsonl jobs found in {input_dir}")
        return 0

    glossary_path = dirs["root"] / schema.SUBDIR / "glossary.json"
    glossary_entries = schema.validate_glossary_file(glossary_path) if glossary_path.is_file() else []
    for jf in job_files:
        schema.verify_manifest(jf, wait_sec=args.manifest_wait)
        schema.validate_jsonl_file(jf)
    print(f"[runner] Pre-validation passed for {len(job_files)} job(s)")

    proc: subprocess.Popen[Any] | None = None
    try:
        if args.api_base:
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

        had_failures = False
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
            )
            if summary.get("failures", 0) or summary.get("missing", 0):
                had_failures = True
        return 1 if had_failures else 0
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
