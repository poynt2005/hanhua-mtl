"""Colab L4 one-click translation runner for Sakura-GalTransl-14B-v3.8.

Enforces:
- Drive-cached llama.cpp b12019 CUDA server (no GLIBC mismatch, zero Docker needed)
- Local SSD model staging (never mmap Google Drive FUSE)
- 4 hard GPU gates (L4/24GB+ VRAM, CUDA0 device, full layer offload, >=12GB VRAM, Gate >=8 tok/s)
- Deduplication, 4-way parallel batching, empty history for ui/system
- Official Sakura-GalTransl v3.8 prompt + sampling + outer/inner speaker stripping
- t2s canonical Simplified normalization (no Traditional retry) + s2t default conversion
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

MODEL_SPEC = schema.MODELS["sakura"]
MODEL_ALIAS = MODEL_SPEC["alias"]
CTX_SIZE = 16384
N_PARALLEL = 4
MIN_TOTAL_VRAM_MIB = 22000.0
MIN_USED_VRAM_MIB = 12000.0
DEFAULT_MIN_TPS = 8.0

SYSTEM_PROMPT = (
    "你是一个视觉小说翻译模型，可以通顺地使用给定的术语表以指定的风格将日文翻译成简体中文，"
    "并联系上下文正确使用人称代词，注意不要混淆使役态和被动态的主语和宾语，"
    "不要擅自添加原文中没有的特殊符号，也不要擅自增加或减少换行。"
)

GATE_SAMPLES = [
    {"id": "gate-1", "kind": "dialogue", "speaker": "ユイ", "text": "おはよう。よく眠れた？"},
    {"id": "gate-2", "kind": "narration", "speaker": None, "text": "\u3000静かな朝の日差しが部屋に差し込んだ。"},
    {"id": "gate-3", "kind": "dialogue", "speaker": "ユイ", "text": "あっ……ダメ、そこは敏感なんだから……♥"},
]

_JA_PRESENT_RE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]")
_KANA_RE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_TEMPLATE_LEAK_RE = re.compile(
    r"(<\|im_start\|>|<\|im_end\|>|</?think[^>]*>|历史翻译：|参考以下术语表|根据以上术语表)"
)
_GENERIC_SPEAKER_RE = re.compile(r"^[^「」『』：:\n]{1,20}[：:]\s*")
_OFFLOAD_RE = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers\s+to\s+GPU", re.I)
_SPACED_PH_RE = re.compile(r"\u27e6\s*(\d+)\s*\u27e7")
_PUNCT_STRIP_RE = re.compile(r"[「」『』\"'‘’“”（）()\[\]…・！？!?、。，．—─ー~～]+")
_PURE_NUM_SYM_RE = re.compile(r"^[0-9.,:;/+\-_%#]+$")
_ASCII_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*")
REPAIR_FLAG_PREFIXES = ("repair_", "trim_", "add_", "heart_norm", "glossary_subst")

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


def to_simplified(text: str) -> str:
    return _opencc("t2s").convert(text) if text else text


def to_traditional(text: str, mode: str = "s2t") -> str:
    if mode not in ("s2t", "s2twp"):
        raise ValueError(f"Unsupported tw_mode {mode!r}; expected 's2t' or 's2twp'")
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
    temperature: float = 0.3,
    top_p: float = 0.8,
    max_tokens: int = 2048,
    timeout: float = 180.0,
    retries: int = 3,
) -> tuple[str, int, float]:
    """Streaming SSE chat completion. Returns (content, approx_tokens, elapsed_sec)."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": 1.0,
        "max_tokens": max_tokens,
        "stream": True,
    }
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
            "LLAMA_ARG_UI": "0",
        }
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", newline="\n") as log_fh:
        proc = subprocess.Popen(
            [str(llama_exe), "--device", "CUDA0", "--cont-batching", "-lv", "4"],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )

    api_base = "http://127.0.0.1:11434/v1"
    deadline = time.monotonic() + 240.0
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


def _prepare_glossary_simplified(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for ent in entries:
        out.append(
            {
                "src": ent["src"],
                "dst": to_simplified(ent["dst"]),
                "note": to_simplified(ent.get("note", "")),
            }
        )
    return out


def _match_glossary(batch_units: list[dict[str, Any]], glossary_cn: list[dict[str, str]]) -> list[dict[str, str]]:
    joined = "\n".join(
        f"{u.get('speaker') or ''} {u['text']}" for u in batch_units
    )
    hits = [ent for ent in glossary_cn if ent["src"] in joined]
    return hits[:50]


def make_unit_key(kind: str, speaker: str | None, text: str) -> str:
    """Stable SHA-1 identity independent of sequential display wid."""
    raw = f"{kind}\0{speaker or ''}\0{text}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def encode_placeholders(text: str) -> tuple[str, list[str]]:
    """Normalize {{TAG_n}} / ⟦n⟧ / \\Cmd[n] tokens to sequential ⟦0⟧, ⟦1⟧, ..."""
    tokens: list[str] = []
    tok_to_idx: dict[str, int] = {}

    def _sub(m: re.Match[str]) -> str:
        tok = m.group(0)
        if tok not in tok_to_idx:
            tok_to_idx[tok] = len(tokens)
            tokens.append(tok)
        return f"\u27e6{tok_to_idx[tok]}\u27e7"

    encoded = schema.PLACEHOLDER_RE.sub(_sub, text or "")
    return encoded, tokens


def decode_placeholders(text: str, tokens: list[str]) -> tuple[str, list[str]]:
    """Restore original placeholder tokens from ⟦i⟧ markers."""
    if not tokens:
        return text, []
    flags: list[str] = []
    normalized = _SPACED_PH_RE.sub(lambda m: f"\u27e6{m.group(1)}\u27e7", text)
    if normalized != text:
        flags.append("repair_placeholder")

    def _restore(m: re.Match[str]) -> str:
        idx = int(m.group(1))
        if 0 <= idx < len(tokens):
            return tokens[idx]
        return m.group(0)

    return _SPACED_PH_RE.sub(_restore, normalized), flags


def _format_source_line(unit: dict[str, Any]) -> tuple[str, list[str]]:
    encoded_text, tokens = encode_placeholders(unit["text"])
    spk = unit.get("speaker")
    if unit["kind"] == "dialogue" and spk:
        return f"{spk}：{encoded_text}", tokens
    return encoded_text, tokens


def build_sakura_messages(
    batch_units: list[dict[str, Any]],
    glossary_cn: list[dict[str, str]],
    history_lines: list[str],
    extra_system_hint: str = "",
) -> list[dict[str, str]]:
    is_ui_only = all(u["kind"] in ("ui", "system") for u in batch_units)
    hist_block = "\n".join(history_lines[-4:]) if (history_lines and not is_ui_only) else ""
    matched = _match_glossary(batch_units, glossary_cn)
    term_lines = []
    for ent in matched:
        line = f"{ent['src']}->{ent['dst']}"
        if ent.get("note"):
            line += f" #{ent['note']}"
        term_lines.append(line)

    src_lines: list[str] = []
    max_ph = 0
    for u in batch_units:
        formatted, tokens = _format_source_line(u)
        src_lines.append(formatted)
        if len(tokens) > max_ph:
            max_ph = len(tokens)
    for idx in range(max_ph):
        term_lines.append(f"\u27e6{idx}\u27e7->\u27e6{idx}\u27e7 #占位符，原样保留")

    parts = [
        f"历史翻译：\n{hist_block}",
        "参考以下术语表（可为空，格式为src->dst #备注）：\n" + "\n".join(term_lines),
        "根据以上术语表的对应关系和备注，结合历史剧情和上下文，将下面的文本从日文翻译成简体中文：\n"
        + "\n".join(src_lines),
    ]
    sys_content = f"{SYSTEM_PROMPT} {extra_system_hint}" if extra_system_hint else SYSTEM_PROMPT
    return [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def strip_speaker_prefix(line: str, unit: dict[str, Any], known_speakers_desc: list[str]) -> str:
    if unit["kind"] != "dialogue" or not unit.get("speaker"):
        return line
    src_spk = unit["speaker"]
    s = line
    for cand in [src_spk] + known_speakers_desc:
        if not cand:
            continue
        for sep in ("：", ":"):
            prefix = cand + sep
            if s.startswith(prefix):
                return s[len(prefix):].lstrip()
    m = _GENERIC_SPEAKER_RE.match(s)
    if m:
        return s[m.end():].lstrip()
    return s


def strip_inner_speaker(line: str, unit: dict[str, Any], known_speakers_desc: list[str]) -> str:
    if unit["kind"] != "dialogue" or not unit.get("speaker"):
        return line
    s = line
    if s.startswith("「"):
        inner = s[1:]
        stripped = strip_speaker_prefix(inner, unit, known_speakers_desc)
        if stripped != inner:
            if stripped.startswith("「") and stripped.endswith("」」"):
                stripped = stripped[:-1]
            elif not stripped.startswith("「"):
                stripped = "「" + stripped
            s = stripped
    if s.endswith("」」") and unit["text"].count("」") < s.count("」"):
        s = s[:-1]
    return s


def repair_quotes(source: str, output: str) -> tuple[str, list[str]]:
    """Deterministically repair quote count/style drift without rewriting text semantics."""
    src_o = source.count("「")
    src_c = source.count("」")
    if src_o == 0 and src_c == 0:
        return output, []

    s = output
    flags: list[str] = []

    if "\uff62" not in source and "\uff63" not in source and ("\uff62" in s or "\uff63" in s):
        s2 = s.replace("\uff62", "「").replace("\uff63", "」")
        if s2 != s:
            s = s2
            flags.append("repair_halfwidth_quotes")

    if src_o == src_c and (s.count("「") < src_o or s.count("」") < src_c):
        deficit_o = src_o - s.count("「")
        deficit_c = src_c - s.count("」")
        if deficit_o == deficit_c and deficit_o > 0:
            if (
                "“" not in source
                and "”" not in source
                and s.count("“") == deficit_o
                and s.count("”") == deficit_c
            ):
                s = s.replace("“", "「").replace("”", "」")
                flags.append("repair_curly_quotes")
            elif '"' not in source and s.count('"') == 2 * deficit_o:
                chars: list[str] = []
                is_open = True
                for ch in s:
                    if ch == '"':
                        chars.append("「" if is_open else "」")
                        is_open = not is_open
                    else:
                        chars.append(ch)
                s = "".join(chars)
                flags.append("repair_ascii_quotes")
            elif (
                "『" not in source
                and "』" not in source
                and s.count("『") == deficit_o
                and s.count("』") == deficit_c
            ):
                s = s.replace("『", "「").replace("』", "」")
                flags.append("repair_double_brackets")

    if s.count("「") == src_o and s.count("」") == src_c:
        return s, flags

    while s.startswith("「「") and s.count("「") > src_o:
        s = s[1:]
        if "trim_dup_open_quote" not in flags:
            flags.append("trim_dup_open_quote")
    while s.endswith("」」") and s.count("」") > src_c:
        s = s[:-1]
        if "trim_dup_close_quote" not in flags:
            flags.append("trim_dup_close_quote")
    if (
        s.startswith("「")
        and s.endswith("」")
        and s.count("「") == src_o + 1
        and s.count("」") == src_c + 1
    ):
        s = s[1:-1]
        flags.append("trim_outer_quote_pair")

    out_o = s.count("「")
    out_c = s.count("」")
    if out_o == src_o and out_c == src_c:
        return s, flags

    if (
        src_o == 1
        and src_c == 0
        and source.startswith("「")
        and out_o == 1
        and out_c == 1
        and s.startswith("「")
        and s.endswith("」")
    ):
        s = s[:-1]
        flags.append("trim_close_quote")
    elif (
        src_o == 0
        and src_c == 1
        and source.endswith("」")
        and out_o == 1
        and out_c == 1
        and s.startswith("「")
        and s.endswith("」")
    ):
        s = s[1:]
        flags.append("trim_open_quote")
    elif src_o == 1 and src_c == 1 and source.startswith("「") and source.endswith("」"):
        if out_o == 0 and out_c == 0:
            s = f"「{s}」"
            flags.append("add_quote_pair")
        elif out_o == 1 and out_c == 0 and s.startswith("「"):
            s = f"{s}」"
            flags.append("add_close_quote")
        elif out_o == 0 and out_c == 1 and s.endswith("」"):
            s = f"「{s}"
            flags.append("add_open_quote")

    return s, flags


def normalize_output_line(
    raw_line: str,
    unit: dict[str, Any],
    known_speakers_desc: list[str],
    matched_glossary: list[dict[str, str]] | None = None,
) -> tuple[str, str | None, list[str]]:
    """Strip mirrored speaker prefixes, repair quotes/hearts/kana glossary, restore indent, normalize t2s."""
    s = raw_line.strip()
    s = strip_speaker_prefix(s, unit, known_speakers_desc)
    s = strip_inner_speaker(s, unit, known_speakers_desc)
    src_text = unit["text"]
    s, flags = repair_quotes(src_text, s)

    if "♥" in src_text and "♡" not in src_text and "♡" in s and "♥" not in s:
        s = s.replace("♡", "♥")
        flags.append("heart_norm")
    elif "♡" in src_text and "♥" not in src_text and "♥" in s and "♡" not in s:
        s = s.replace("♥", "♡")
        flags.append("heart_norm")

    if _KANA_RE.search(s) and matched_glossary:
        for ent in sorted(matched_glossary, key=lambda e: len(e["src"]), reverse=True):
            if _KANA_RE.search(ent["src"]) and ent["src"] in s:
                s = s.replace(ent["src"], ent["dst"])
                if "glossary_subst" not in flags:
                    flags.append("glossary_subst")

    if src_text.startswith("\u3000") and not s.startswith("\u3000"):
        s = "\u3000" + s.lstrip()
    canonical = to_simplified(s)
    raw_saved: str | None = None
    if canonical != s:
        flags.append("t2s")
        raw_saved = s
    return canonical, raw_saved, flags


def is_allowlisted_no_cjk(
    src_text: str,
    zh_cn: str,
    allowlist: tuple[re.Pattern[str], ...] = (),
) -> bool:
    """Allow pure numbers or ASCII identifiers/commands that appeared verbatim in source."""
    no_ph = schema.PLACEHOLDER_RE.sub("", zh_cn or "")
    core = _PUNCT_STRIP_RE.sub("", no_ph).strip()
    if not core:
        return False
    if _PURE_NUM_SYM_RE.fullmatch(core):
        return True
    for pat in allowlist:
        if pat.search(core):
            return True
    words = _ASCII_WORD_RE.findall(core)
    if words:
        src_lower = src_text.lower()
        return all(w.lower() in src_lower for w in words)
    return False


def check_language_qa(
    src_text: str,
    zh_cn: str,
    allowlist: tuple[re.Pattern[str], ...] = (),
) -> list[str]:
    """Ensure Japanese-bearing units actually produce Chinese (no CJK / Latin-only failures)."""
    if not _JA_PRESENT_RE.search(src_text or ""):
        return []
    if _CJK_RE.search(zh_cn or ""):
        return []
    if is_allowlisted_no_cjk(src_text, zh_cn, allowlist):
        return []
    errs = ["no_cjk"]
    stripped_ph = schema.PLACEHOLDER_RE.sub("", zh_cn or "")
    if re.search(r"[A-Za-z]", stripped_ph):
        errs.append("latin_only")
    return errs


def validate_unit_translation(
    src_text: str,
    zh_cn: str,
    allowlist: tuple[re.Pattern[str], ...] = (),
) -> list[str]:
    errs: list[str] = []
    if not zh_cn or not zh_cn.strip():
        errs.append("empty")
        return errs
    if _TEMPLATE_LEAK_RE.search(zh_cn):
        errs.append("template_leak")
    if _KANA_RE.search(zh_cn):
        errs.append("residual_kana")
    if schema.extract_placeholders(src_text) != schema.extract_placeholders(zh_cn):
        errs.append("placeholder_mismatch")
    for ch in ("♥", "♡"):
        if src_text.count(ch) != zh_cn.count(ch):
            errs.append(f"heart_mismatch:{ch}")
    if ("「" in src_text or "」" in src_text) and (
        src_text.count("「") != zh_cn.count("「") or src_text.count("」") != zh_cn.count("」")
    ):
        errs.append("quote_asymmetry_mismatch")
    errs.extend(check_language_qa(src_text, zh_cn, allowlist=allowlist))
    return errs


def run_preflight_gate(api_base: str, min_tps: float = DEFAULT_MIN_TPS) -> None:
    print("[gate] Running 3-line Sakura pre-flight gate...")
    total_tokens = 0
    total_sec = 0.0
    for idx, sample in enumerate(GATE_SAMPLES, start=1):
        msgs = build_sakura_messages([sample], [], [])
        last_err: str | None = None
        cn = ""
        for temp in (0.3, 0.1, 0.0):
            out, toks, elapsed = chat_stream(
                api_base, MODEL_ALIAS, msgs, temperature=temp, top_p=0.8, max_tokens=512
            )
            lines = [ln for ln in out.splitlines() if ln.strip()]
            if len(lines) != 1:
                last_err = f"Gate sample {idx} returned {len(lines)} lines: {out!r}"
                continue
            cand, _, _ = normalize_output_line(lines[0], sample, ["ユイ"])
            errs = validate_unit_translation(sample["text"], cand)
            if errs:
                last_err = f"Gate sample {idx} failed validation {errs}: {cand!r}"
                continue
            cn = cand
            total_tokens += toks
            total_sec += elapsed
            last_err = None
            break
        if last_err is not None:
            raise RuntimeError(last_err)
        print(f"  [gate {idx}/3 OK] {sample['text']} -> {cn}")

    tps = total_tokens / max(0.001, total_sec)
    print(f"[gate] Gate 4 PASS: {tps:.1f} tok/s (threshold >= {min_tps:.1f})")
    if tps < min_tps:
        raise RuntimeError(f"Gate 4 failed: throughput {tps:.1f} tok/s < {min_tps:.1f} (CPU fallback?)")


def needs_translation(rec: dict[str, Any]) -> bool:
    if rec.get("override") is not None:
        return False
    return bool(_JA_PRESENT_RE.search(rec["text"]))


def _split_batches(units: list[dict[str, Any]], max_batch: int = 8) -> list[list[dict[str, Any]]]:
    """Group into batches <= max_batch, never mixing ui/system with story or batching multiline rows."""
    batches: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    cur_ui: bool | None = None
    for u in units:
        is_ui = u["kind"] in ("ui", "system")
        has_nl = "\n" in u["text"]
        if has_nl:
            if cur:
                batches.append(cur)
                cur = []
                cur_ui = None
            batches.append([u])
            continue
        if cur and (len(cur) >= max_batch or cur_ui != is_ui):
            batches.append(cur)
            cur = []
        cur.append(u)
        cur_ui = is_ui
    if cur:
        batches.append(cur)
    return batches


def _hint_for_errors(errs: list[str]) -> str:
    if "placeholder_mismatch" in errs:
        return "必须原样保留所有占位符标记（如⟦0⟧、⟦1⟧），不要删除、翻译、重复或调换顺序。"
    if "quote_asymmetry_mismatch" in errs:
        return "请严格保留原文中「和」的数量与位置，不要将「」替换为其他引号，也不要省略。"
    if "residual_kana" in errs:
        return "请将所有日文假名（包括外来语、专有名词、人名与道具名）完整翻译或音译为简体中文，禁止在译文中保留任何平假名或片假名。"
    if "no_cjk" in errs or "latin_only" in errs:
        return "必须完整输出简体中文译文，禁止输出英文单词或省略正文。"
    return ""


def _build_failure_record(
    unit: dict[str, Any],
    errs: list[str],
    attempts: int,
    observed_text: str,
    glossary_cn: list[dict[str, str]],
) -> dict[str, Any]:
    expected_ph = schema.extract_placeholders(unit["text"])
    observed_ph = schema.extract_placeholders(observed_text)
    suggested = "override"
    if "residual_kana" in errs:
        matched = _match_glossary([unit], glossary_cn)
        if matched:
            suggested = "override:" + ",".join(f"{e['src']}->{e['dst']}" for e in matched)
        else:
            suggested = "glossary_or_override"
    elif "placeholder_mismatch" in errs:
        suggested = "override_keep_tags"
    elif "quote_asymmetry_mismatch" in errs:
        suggested = "override_keep_quotes"
    return {
        "unit_key": unit["unit_key"],
        "wid": unit["wid"],
        "text": unit["text"],
        "errors": errs,
        "attempts": attempts,
        "expected_placeholders": expected_ph,
        "observed_placeholders": observed_ph,
        "suggested_fix": suggested,
        "needs_override": True,
        "source_kind": unit["kind"],
    }


def _process_model_line(
    raw_line: str,
    unit: dict[str, Any],
    tokens: list[str],
    known_speakers_desc: list[str],
    matched_glossary: list[dict[str, str]],
) -> tuple[str, str | None, list[str], list[str]]:
    decoded, ph_flags = decode_placeholders(raw_line, tokens)
    cn, raw_saved, flags = normalize_output_line(
        decoded, unit, known_speakers_desc, matched_glossary=matched_glossary
    )
    all_flags = ph_flags + flags
    errs = validate_unit_translation(unit["text"], cn)
    return cn, raw_saved, all_flags, errs


def _translate_single_unit(
    api_base: str,
    unit: dict[str, Any],
    glossary_cn: list[dict[str, str]],
    history: list[str],
    known_speakers_desc: list[str],
) -> tuple[str | None, str | None, list[str], dict[str, Any] | None]:
    matched = _match_glossary([unit], glossary_cn)
    _, tokens = encode_placeholders(unit["text"])
    schedule = ((0.3, True), (0.1, True), (0.0, False))
    last_err: list[str] = ["unknown"]
    last_obs = ""
    attempts = 0
    extra_hint = ""
    for idx, (temp, use_hist) in enumerate(schedule):
        attempts = idx + 1
        msgs = build_sakura_messages(
            [unit],
            glossary_cn,
            history if use_hist else [],
            extra_system_hint=extra_hint,
        )
        try:
            raw, _, _ = chat_stream(
                api_base, MODEL_ALIAS, msgs, temperature=temp, top_p=0.8, max_tokens=512
            )
        except Exception as exc:
            last_err = [f"transport:{exc}"]
            continue
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        if len(lines) != 1:
            last_err = [f"line_count:{len(lines)}!=1"]
            last_obs = raw
            continue
        cn, raw_saved, flags, errs = _process_model_line(
            lines[0], unit, tokens, known_speakers_desc, matched
        )
        if not errs:
            return cn, raw_saved, flags, None
        last_err = errs
        last_obs = cn
        extra_hint = _hint_for_errors(errs)
    return None, None, [], _build_failure_record(unit, last_err, attempts, last_obs, glossary_cn)


def _translate_single_multiline(
    api_base: str,
    unit: dict[str, Any],
    glossary_cn: list[dict[str, str]],
    history: list[str],
    known_speakers_desc: list[str],
) -> tuple[str | None, str | None, list[str], dict[str, Any] | None]:
    raw_segments = unit["text"].split("\n")
    out_segments: list[str] = []
    raw_segments_out: list[str] = []
    agg_flags: list[str] = []
    seg_hist = list(history)
    total_attempts = 0
    saw_first_nonempty = False

    for seg in raw_segments:
        if not seg.strip():
            out_segments.append(seg)
            raw_segments_out.append(seg)
            continue
        seg_unit = {
            "unit_key": unit["unit_key"],
            "wid": unit["wid"],
            "file": unit["file"],
            "kind": unit["kind"],
            "speaker": unit.get("speaker") if not saw_first_nonempty else None,
            "text": seg,
        }
        saw_first_nonempty = True
        cn_seg, raw_seg, seg_flags, fail_rec = _translate_single_unit(
            api_base, seg_unit, glossary_cn, seg_hist, known_speakers_desc
        )
        if fail_rec is not None or cn_seg is None:
            total_attempts += fail_rec["attempts"] if fail_rec else 1
            errs = ["multiline_failed"] + (fail_rec["errors"] if fail_rec else [])
            obs = "\n".join(out_segments)
            return None, None, [], _build_failure_record(unit, errs, total_attempts, obs, glossary_cn)
        total_attempts += 1
        out_segments.append(cn_seg)
        raw_segments_out.append(raw_seg if raw_seg is not None else cn_seg)
        for fl in seg_flags:
            if fl not in agg_flags:
                agg_flags.append(fl)
        if unit["kind"] not in ("ui", "system"):
            seg_hist = (seg_hist + [cn_seg])[-4:]

    if len(out_segments) != len(raw_segments):
        return (
            None,
            None,
            [],
            _build_failure_record(
                unit,
                ["multiline_failed", "segment_count_mismatch"],
                total_attempts,
                "\n".join(out_segments),
                glossary_cn,
            ),
        )

    joined = "\n".join(out_segments)
    errs = validate_unit_translation(unit["text"], joined)
    if errs:
        return (
            None,
            None,
            [],
            _build_failure_record(unit, ["multiline_failed"] + errs, total_attempts, joined, glossary_cn),
        )
    raw_joined = (
        "\n".join(raw_segments_out)
        if any(r != c for r, c in zip(raw_segments_out, out_segments))
        else None
    )
    return joined, raw_joined, agg_flags, None


def _translate_batch_recursive(
    api_base: str,
    batch: list[dict[str, Any]],
    glossary_cn: list[dict[str, str]],
    history: list[str],
    known_speakers_desc: list[str],
) -> tuple[dict[str, tuple[str, str | None, list[str]]], list[dict[str, Any]], list[str]]:
    """Returns (succeeded_map_by_unit_key, failed_records, updated_history)."""
    if len(batch) == 1:
        u = batch[0]
        if "\n" in u["text"]:
            cn, raw_saved, flags, fail_rec = _translate_single_multiline(
                api_base, u, glossary_cn, history, known_speakers_desc
            )
        else:
            cn, raw_saved, flags, fail_rec = _translate_single_unit(
                api_base, u, glossary_cn, history, known_speakers_desc
            )
        if fail_rec is None and cn is not None:
            new_hist = history if u["kind"] in ("ui", "system") else (history + [cn])[-4:]
            return {u["unit_key"]: (cn, raw_saved, flags)}, [], new_hist
        assert fail_rec is not None
        return {}, [fail_rec], history

    matched = _match_glossary(batch, glossary_cn)
    unit_tokens = [encode_placeholders(u["text"])[1] for u in batch]
    for temp in (0.3, 0.1):
        msgs = build_sakura_messages(batch, glossary_cn, history)
        try:
            raw, _, _ = chat_stream(
                api_base, MODEL_ALIAS, msgs, temperature=temp, top_p=0.8, max_tokens=2048
            )
        except Exception:
            continue

        lines = [ln for ln in raw.splitlines() if ln.strip()]
        if len(lines) != len(batch):
            continue

        parsed: dict[str, tuple[str, str | None, list[str]]] = {}
        batch_ok = True
        new_hist = list(history)
        for u, tokens, ln in zip(batch, unit_tokens, lines):
            cn, raw_saved, flags, errs = _process_model_line(
                ln, u, tokens, known_speakers_desc, matched
            )
            if errs:
                batch_ok = False
                break
            parsed[u["unit_key"]] = (cn, raw_saved, flags)
            if u["kind"] not in ("ui", "system"):
                new_hist.append(cn)
        if batch_ok:
            return parsed, [], new_hist[-4:]

    mid = len(batch) // 2
    left_ok, left_fail, hist_mid = _translate_batch_recursive(
        api_base, batch[:mid], glossary_cn, history, known_speakers_desc
    )
    right_ok, right_fail, hist_end = _translate_batch_recursive(
        api_base, batch[mid:], glossary_cn, hist_mid, known_speakers_desc
    )
    merged = dict(left_ok)
    merged.update(right_ok)
    return merged, left_fail + right_fail, hist_end


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
    tw_mode: str = "s2t",
    force: bool = False,
    manifest_wait: float = 60.0,
    skip_gpu_check: bool = False,
    overrides_path: Path | None = None,
    repair_only: bool = False,
) -> dict[str, Any]:
    try:
        if not force and not repair_only and schema.done_marker_path(output_dir, job).is_file():
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

    # Known speakers longest-first for accurate prefix stripping
    speaker_set: set[str] = set()
    for ent in glossary_cn:
        speaker_set.add(ent["src"])
        speaker_set.add(ent["dst"])
    for rec in records:
        if rec.get("speaker"):
            speaker_set.add(str(rec["speaker"]))
    known_speakers_desc = sorted(speaker_set, key=len, reverse=True)

    # First pass without external overrides builds exact legacy sequential wid -> unit_key map
    legacy_wid_to_key: dict[str, str] = {}
    legacy_seen: set[str] = set()
    for rec in records:
        if rec.get("override") is not None or not needs_translation(rec):
            continue
        uk = make_unit_key(rec["kind"], rec.get("speaker"), rec["text"])
        if uk not in legacy_seen:
            legacy_seen.add(uk)
            legacy_wid_to_key[f"w{len(legacy_seen):06d}"] = uk

    # Deduplicate translatable records by stable unit_key = sha1(kind\0speaker\0text)
    work_units: list[dict[str, Any]] = []
    seen_keys: dict[str, dict[str, Any]] = {}
    rec_to_unit_key: dict[str, str] = {}
    override_cn: dict[str, str] = {}
    passthrough_cn: dict[str, str] = {}

    for rec in records:
        rid = rec["id"]
        if rec.get("override") is not None:
            override_cn[rid] = to_simplified(str(rec["override"]))
            continue
        if rid in ov_by_id:
            override_cn[rid] = to_simplified(ov_by_id[rid])
            continue
        uk = make_unit_key(rec["kind"], rec.get("speaker"), rec["text"])
        if uk in ov_by_key:
            override_cn[rid] = to_simplified(ov_by_key[uk])
            continue
        if not needs_translation(rec):
            passthrough_cn[rid] = rec["text"]
            continue
        if uk not in seen_keys:
            wid = f"w{len(work_units) + 1:06d}"
            unit_obj = {
                "unit_key": uk,
                "wid": wid,
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

    todo_units = [u for u in work_units if u["unit_key"] not in done_map]
    cached_count = len(work_units) - len(todo_units)
    print(
        f"[job] {job}: total={len(records)} | passthrough={len(passthrough_cn)} | "
        f"overrides={len(override_cn)} | unique_work={len(work_units)} | "
        f"cached={cached_count} | todo={len(todo_units)}"
    )

    # Group todo units by file; if single large file, split into 4 chunks to saturate -np 4
    by_file: dict[str, list[dict[str, Any]]] = {}
    for u in todo_units:
        by_file.setdefault(u["file"], []).append(u)
    queues: list[list[dict[str, Any]]] = list(by_file.values())
    if len(queues) == 1 and len(queues[0]) > 32:
        items = queues[0]
        step = max(8, (len(items) + N_PARALLEL - 1) // N_PARALLEL)
        queues = [items[i : i + step] for i in range(0, len(items), step)]

    ckpt_lock = threading.Lock()
    gpu_log_lock = threading.Lock()
    progress_counter = {"count": cached_count}
    all_failures: list[dict[str, Any]] = []

    def _run_queue(q_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
        q_failures: list[dict[str, Any]] = []
        hist: list[str] = []
        for batch in _split_batches(q_units, max_batch=8):
            ok_map, fails, hist = _translate_batch_recursive(
                api_base, batch, glossary_cn, hist, known_speakers_desc
            )
            if ok_map:
                should_log_gpu = False
                cur_done = 0
                with ckpt_lock:
                    with open(ckpt_path, "a", encoding="utf-8", newline="\n") as fh:
                        for uk, (cn, raw_saved, flags) in ok_map.items():
                            row: dict[str, Any] = {
                                "unit_key": uk,
                                "wid": seen_keys[uk]["wid"],
                                "zh_cn": cn,
                            }
                            if raw_saved is not None:
                                row["raw"] = raw_saved
                            if flags:
                                row["flags"] = flags
                            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                            done_map[uk] = row
                        fh.flush()
                    prev = progress_counter["count"]
                    progress_counter["count"] += len(ok_map)
                    cur_done = progress_counter["count"]
                    should_log_gpu = (cur_done // 50) > (prev // 50)
                if should_log_gpu:
                    with gpu_log_lock:
                        _log_gpu_progress(
                            cur_done, len(work_units), gpu_log, skip_gpu_check
                        )
            q_failures.extend(fails)
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

    # Re-load checkpoint cleanly and expand in exact original record order
    final_done, _ = _load_checkpoint(ckpt_path, legacy_wid_to_key=legacy_wid_to_key)
    missing_ids: list[str] = []
    cn_rows: list[dict[str, Any]] = []
    tw_rows: list[dict[str, Any]] = []
    zh_cn_texts: list[str] = []
    zh_tw_texts: list[str] = []
    no_cjk_count = 0
    latin_only_count = 0
    repaired_units: set[str] = set()

    for rec in records:
        rid = rec["id"]
        if rid in override_cn:
            cn_val = override_cn[rid]
        elif rid in passthrough_cn:
            cn_val = passthrough_cn[rid]
        else:
            uk = rec_to_unit_key[rid]
            entry = final_done.get(uk)
            if not entry:
                missing_ids.append(rid)
                cn_val = ""
            else:
                cn_val = entry["zh_cn"]
                flags_list = entry.get("flags") or []
                if any(str(fl).startswith(REPAIR_FLAG_PREFIXES) for fl in flags_list):
                    repaired_units.add(uk)
                lang_errs = check_language_qa(rec["text"], cn_val)
                if "no_cjk" in lang_errs:
                    no_cjk_count += 1
                if "latin_only" in lang_errs:
                    latin_only_count += 1
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

    roundtrip_mismatches = 0
    if tw_mode == "s2t":
        for cn_s, tw_s in zip(zh_cn_texts, zh_tw_texts):
            if to_simplified(tw_s) != cn_s:
                roundtrip_mismatches += 1

    qa_summary = {
        "job": job,
        "total_records": len(records),
        "unique_work_units": len(work_units),
        "passthrough_records": len(passthrough_cn),
        "manual_overrides": len(override_cn),
        "repaired": len(repaired_units),
        "missing": len(missing_ids),
        "failures": len(all_failures),
        "no_cjk": no_cjk_count,
        "latin_only": latin_only_count,
        "tw_mode": tw_mode,
        "roundtrip_mismatches": roundtrip_mismatches,
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

    if not missing_ids and not all_failures and no_cjk_count == 0 and latin_only_count == 0:
        marker = schema.build_done_marker(job, [cn_out, tw_out, qa_out], qa_summary)
        schema.atomic_write_json(schema.done_marker_path(output_dir, job), marker)
        print(f"[job] {job}: COMPLETE 100% ({len(records)}/{len(records)}) -> {tw_out.name}")
    else:
        print(
            f"[job] {job}: INCOMPLETE missing={len(missing_ids)} failures={len(all_failures)} "
            f"no_cjk={no_cjk_count} latin_only={latin_only_count}; DONE marker NOT written"
        )
    return qa_summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Colab L4 Sakura translation runner")
    ap.add_argument("--drive-root", default=f"/content/drive/MyDrive/{schema.DRIVE_ROOT_NAME}")
    ap.add_argument("--api-base", help="Use an already-running OpenAI-compatible /v1 endpoint")
    ap.add_argument("--tw-mode", default="s2t", choices=("s2t", "s2twp"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--repair-only", action="store_true", help="Resume from checkpoint and repair remaining units only")
    ap.add_argument("--job", help="Run only the named job (<job>.jsonl)")
    ap.add_argument("--overrides", help="Optional path to overrides JSONL file")
    ap.add_argument("--skip-gpu-check", action="store_true")
    ap.add_argument("--manifest-wait", type=float, default=60.0)
    ap.add_argument("--min-tps", type=float, default=DEFAULT_MIN_TPS)
    args = ap.parse_args(argv)

    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")

    dirs = schema.ensure_drive_tree(args.drive_root)
    input_dir = dirs["sakura_input"]
    work_dir = dirs["sakura_work"]
    output_dir = dirs["sakura_output"]

    # Strict check: reject non-.jsonl/.manifest.json/.overrides.jsonl files inside input_dir (ignore OS metadata)
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
        return 0

    overrides_path = Path(args.overrides) if args.overrides else None

    # Pre-validate ALL jobs and optional glossary before touching GPU/server
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
            # Gate 0: fail fast before staging prebuilt binary or copying 12GB model if T4/CPU
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
                overrides_path=overrides_path,
                repair_only=args.repair_only,
            )
            if (
                summary.get("failures", 0)
                or summary.get("missing", 0)
                or summary.get("no_cjk", 0)
                or summary.get("latin_only", 0)
            ):
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
