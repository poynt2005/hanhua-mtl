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
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import schema  # noqa: E402

LLAMA_TAG = "b12019"
LLAMA_REPO = "https://github.com/ggml-org/llama.cpp.git"
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


def _env_cache_hash(llama_tag: str = LLAMA_TAG) -> str:
    nvcc_out = subprocess.run(
        ["nvcc", "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace"
    ).stdout
    ldd_out = subprocess.run(
        ["ldd", "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace"
    ).stdout.splitlines()[0:1]
    sig = f"{llama_tag}|{nvcc_out.strip()}|{' '.join(ldd_out)}"
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()[:12]


def _bin_env(local_bin_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    ld_prev = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{local_bin_dir}:{ld_prev}" if ld_prev else str(local_bin_dir)
    return env


def ensure_llama_server(
    drive_root: Path,
    local_bin_dir: Path,
    llama_tag: str = LLAMA_TAG,
    force_rebuild: bool = False,
) -> Path:
    local_bin_dir.mkdir(parents=True, exist_ok=True)
    local_exe = local_bin_dir / "llama-server"
    env_hash = _env_cache_hash(llama_tag)
    cache_tar = drive_root / "bin" / f"llama-server-l4-{env_hash}.tar.gz"
    cache_sha = drive_root / "bin" / f"llama-server-l4-{env_hash}.tar.gz.sha256"

    if not force_rebuild and cache_tar.is_file():
        print(f"[bin] Found cached llama-server bundle: {cache_tar.name}")
        ok_sha = True
        if cache_sha.is_file():
            exp = cache_sha.read_text(encoding="utf-8-sig").strip().split()[0]
            ok_sha = schema.sha256_file(cache_tar) == exp
        if ok_sha:
            try:
                with tarfile.open(cache_tar, "r:gz") as tf:
                    tf.extractall(local_bin_dir)
                if local_exe.is_file():
                    local_exe.chmod(0o755)
                    chk = subprocess.run(
                        [str(local_exe), "--version"],
                        env=_bin_env(local_bin_dir),
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                    if chk.returncode == 0:
                        print("[bin] Cached llama-server verified OK")
                        return local_exe
                    print(f"[bin] Cached binary unusable (rc={chk.returncode}), rebuilding...")
            except Exception as exc:
                print(f"[bin] Cache extraction failed ({exc}), rebuilding...")
        else:
            print("[bin] Cache sha256 mismatch, rebuilding...")

    build_src = Path("/content/_llama_src")
    build_dir = build_src / "build"
    if build_src.exists():
        shutil.rmtree(build_src, ignore_errors=True)
    print(f"[bin] Cloning llama.cpp {llama_tag} and building CUDA sm_80/89...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", llama_tag, LLAMA_REPO, str(build_src)],
        check=True,
    )
    subprocess.run(
        [
            "cmake",
            "-S",
            str(build_src),
            "-B",
            str(build_dir),
            "-DGGML_CUDA=ON",
            "-DCMAKE_CUDA_ARCHITECTURES=89;80",
            "-DLLAMA_CURL=OFF",
            "-DBUILD_SHARED_LIBS=ON",
        ],
        check=True,
    )
    nproc = str(max(2, os.cpu_count() or 2))
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "llama-server", "-j", nproc],
        check=True,
    )

    built_exe = build_dir / "bin" / "llama-server"
    if not built_exe.is_file():
        found = next((p for p in build_dir.rglob("llama-server") if p.is_file()), None)
        if found is None:
            raise RuntimeError(f"Built llama-server not found under {build_dir}")
        built_exe = found
    shutil.copy2(built_exe, local_exe)
    local_exe.chmod(0o755)
    for so_file in (build_dir / "bin").glob("*.so*"):
        shutil.copy2(so_file, local_bin_dir / so_file.name)
    for so_file in build_dir.rglob("*.so*"):
        if not (local_bin_dir / so_file.name).exists():
            shutil.copy2(so_file, local_bin_dir / so_file.name)

    tmp_tar = local_bin_dir.parent / f".{cache_tar.name}.tmp"
    with tarfile.open(tmp_tar, "w:gz") as tf:
        for item in local_bin_dir.iterdir():
            tf.add(item, arcname=item.name)
    cache_tar.parent.mkdir(parents=True, exist_ok=True)
    tmp_drive_tar = cache_tar.with_name(f".{cache_tar.name}.tmp")
    shutil.copy2(tmp_tar, tmp_drive_tar)
    os.replace(tmp_drive_tar, cache_tar)
    schema.atomic_write_text(cache_sha, f"{schema.sha256_file(cache_tar)}  {cache_tar.name}\n")
    tmp_tar.unlink(missing_ok=True)
    shutil.rmtree(build_src, ignore_errors=True)
    print(f"[bin] Cached built llama-server to {cache_tar}")
    return local_exe


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
    print(f"[model] Verifying Drive SHA-256 for {drive_model.name}...")
    drive_sha = schema.sha256_file(drive_model)
    if drive_sha != MODEL_SPEC["sha256"]:
        raise RuntimeError(
            f"Drive model SHA-256 mismatch: {drive_sha} != {MODEL_SPEC['sha256']}. "
            "Re-run deploy_models_colab.ipynb."
        )

    local_models_dir.mkdir(parents=True, exist_ok=True)
    local_model = local_models_dir / MODEL_SPEC["filename"]
    if local_model.is_file() and local_model.stat().st_size == MODEL_SPEC["size_bytes"]:
        print(f"[model] Local SSD model already staged: {local_model}")
        return local_model

    tmp_local = local_model.with_name(f".{local_model.name}.tmp")
    print(f"[model] Staging {MODEL_SPEC['size_bytes'] / 1e9:.2f} GB from Drive to local SSD...")
    t0 = time.monotonic()
    shutil.copyfile(drive_model, tmp_local)
    os.replace(tmp_local, local_model)
    print(f"[model] Staged in {time.monotonic() - t0:.1f}s -> {local_model}")
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
            [str(llama_exe), "--device", "CUDA0", "--cont-batching"],
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

    offloaded, total = parse_gpu_offload_log(log_path.read_text(encoding="utf-8", errors="replace"))
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


def _format_source_line(unit: dict[str, Any]) -> str:
    spk = unit.get("speaker")
    if unit["kind"] == "dialogue" and spk:
        return f"{spk}：{unit['text']}"
    return unit["text"]


def build_sakura_messages(
    batch_units: list[dict[str, Any]],
    glossary_cn: list[dict[str, str]],
    history_lines: list[str],
    strict_line_hint: bool = False,
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
    src_lines = [_format_source_line(u) for u in batch_units]

    parts = [
        f"历史翻译：\n{hist_block}",
        "参考以下术语表（可为空，格式为src->dst #备注）：\n" + "\n".join(term_lines),
        "根据以上术语表的对应关系和备注，结合历史剧情和上下文，将下面的文本从日文翻译成简体中文："
        + (f"（请严格输出恰好 {len(src_lines)} 行，一一对应，不要合并或拆分行）" if strict_line_hint else "")
        + "\n"
        + "\n".join(src_lines),
    ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
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


def normalize_output_line(
    raw_line: str,
    unit: dict[str, Any],
    known_speakers_desc: list[str],
) -> tuple[str, str | None, list[str]]:
    """Strip mirrored speaker prefixes, restore full-width indent, normalize t2s (never retry)."""
    s = raw_line.strip()
    s = strip_speaker_prefix(s, unit, known_speakers_desc)
    s = strip_inner_speaker(s, unit, known_speakers_desc)
    if unit["text"].startswith("\u3000") and not s.startswith("\u3000"):
        s = "\u3000" + s.lstrip()
    canonical = to_simplified(s)
    flags: list[str] = []
    raw_saved: str | None = None
    if canonical != s:
        flags.append("t2s")
        raw_saved = s
    return canonical, raw_saved, flags


def validate_unit_translation(src_text: str, zh_cn: str) -> list[str]:
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
    return errs


def run_preflight_gate(api_base: str, min_tps: float = DEFAULT_MIN_TPS) -> None:
    print("[gate] Running 3-line Sakura pre-flight gate...")
    total_tokens = 0
    total_sec = 0.0
    for idx, sample in enumerate(GATE_SAMPLES, start=1):
        msgs = build_sakura_messages([sample], [], [], strict_line_hint=True)
        out, toks, elapsed = chat_stream(
            api_base, MODEL_ALIAS, msgs, temperature=0.3, top_p=0.8, max_tokens=512
        )
        lines = [ln for ln in out.splitlines() if ln.strip()]
        if len(lines) != 1:
            raise RuntimeError(f"Gate sample {idx} returned {len(lines)} lines: {out!r}")
        cn, _, _ = normalize_output_line(lines[0], sample, ["ユイ"])
        errs = validate_unit_translation(sample["text"], cn)
        if errs:
            raise RuntimeError(f"Gate sample {idx} failed validation {errs}: {cn!r}")
        total_tokens += toks
        total_sec += elapsed
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


def _translate_single_multiline(
    api_base: str,
    unit: dict[str, Any],
    glossary_cn: list[dict[str, str]],
    history: list[str],
    known_speakers_desc: list[str],
) -> tuple[str | None, str | None, list[str], list[str]]:
    expected_lines = unit["text"].splitlines()
    for attempt_temp, use_hist in ((0.3, True), (0.1, True), (0.1, False)):
        msgs = build_sakura_messages(
            [unit],
            glossary_cn,
            history if use_hist else [],
            strict_line_hint=True,
        )
        try:
            raw, _, _ = chat_stream(
                api_base, MODEL_ALIAS, msgs, temperature=attempt_temp, top_p=0.8, max_tokens=1024
            )
        except Exception:
            continue
        out_lines = [ln for ln in raw.splitlines() if ln.strip()]
        if len(out_lines) != len(expected_lines):
            continue
        joined = "\n".join(out_lines)
        cn, raw_saved, flags = normalize_output_line(joined, unit, known_speakers_desc)
        errs = validate_unit_translation(unit["text"], cn)
        if not errs:
            return cn, raw_saved, flags, []
    return None, None, [], ["multiline_failed"]


def _translate_batch_recursive(
    api_base: str,
    batch: list[dict[str, Any]],
    glossary_cn: list[dict[str, str]],
    history: list[str],
    known_speakers_desc: list[str],
) -> tuple[dict[str, tuple[str, str | None, list[str]]], list[dict[str, Any]], list[str]]:
    """Returns (succeeded_map, failed_records, updated_history)."""
    if len(batch) == 1 and "\n" in batch[0]["text"]:
        u = batch[0]
        cn, raw_saved, flags, errs = _translate_single_multiline(
            api_base, u, glossary_cn, history, known_speakers_desc
        )
        if cn is not None:
            return {u["wid"]: (cn, raw_saved, flags)}, [], (history + [cn])[-4:]
        return {}, [{"wid": u["wid"], "text": u["text"], "errors": errs}], history

    temps = (0.3, 0.1, 0.1)
    last_err: list[str] = ["unknown"]
    for idx, temp in enumerate(temps):
        use_hist = history if (len(batch) > 1 or idx < 2) else []
        msgs = build_sakura_messages(
            batch, glossary_cn, use_hist, strict_line_hint=(idx > 0)
        )
        try:
            raw, _, _ = chat_stream(
                api_base,
                MODEL_ALIAS,
                msgs,
                temperature=temp,
                top_p=0.8,
                max_tokens=2048 if len(batch) > 1 else 512,
            )
        except Exception as exc:
            last_err = [f"transport:{exc}"]
            continue

        lines = [ln for ln in raw.splitlines() if ln.strip()]
        if len(lines) != len(batch):
            last_err = [f"line_count:{len(lines)}!={len(batch)}"]
            continue

        parsed: dict[str, tuple[str, str | None, list[str]]] = {}
        batch_ok = True
        batch_errs: list[str] = []
        new_hist = list(history)
        for u, ln in zip(batch, lines):
            cn, raw_saved, flags = normalize_output_line(ln, u, known_speakers_desc)
            errs = validate_unit_translation(u["text"], cn)
            if errs:
                batch_ok = False
                batch_errs = errs
                break
            parsed[u["wid"]] = (cn, raw_saved, flags)
            if u["kind"] not in ("ui", "system"):
                new_hist.append(cn)
        if batch_ok:
            return parsed, [], new_hist[-4:]
        last_err = batch_errs

    if len(batch) > 1:
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

    u = batch[0]
    return {}, [{"wid": u["wid"], "text": u["text"], "errors": last_err}], history


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
    tw_mode: str = "s2t",
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

    # Known speakers longest-first for accurate prefix stripping
    speaker_set: set[str] = set()
    for ent in glossary_cn:
        speaker_set.add(ent["src"])
        speaker_set.add(ent["dst"])
    for rec in records:
        if rec.get("speaker"):
            speaker_set.add(str(rec["speaker"]))
    known_speakers_desc = sorted(speaker_set, key=len, reverse=True)

    # Deduplicate translatable records by (kind, speaker, text)
    work_units: list[dict[str, Any]] = []
    key_to_wid: dict[tuple[str, str, str], str] = {}
    rec_to_wid: dict[str, str] = {}
    passthrough_cn: dict[str, str] = {}

    for rec in records:
        if rec.get("override") is not None:
            passthrough_cn[rec["id"]] = to_simplified(str(rec["override"]))
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
    progress_counter = {"count": len(done_map)}
    all_failures: list[dict[str, Any]] = []

    def _run_queue(q_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
        q_failures: list[dict[str, Any]] = []
        hist: list[str] = []
        for batch in _split_batches(q_units, max_batch=8):
            ok_map, fails, hist = _translate_batch_recursive(
                api_base, batch, glossary_cn, hist, known_speakers_desc
            )
            if ok_map:
                with ckpt_lock:
                    with open(ckpt_path, "a", encoding="utf-8", newline="\n") as fh:
                        for wid, (cn, raw_saved, flags) in ok_map.items():
                            row: dict[str, Any] = {"wid": wid, "zh_cn": cn}
                            if raw_saved is not None:
                                row["raw"] = raw_saved
                            if flags:
                                row["flags"] = flags
                            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                            done_map[wid] = row
                        fh.flush()
                    prev = progress_counter["count"]
                    progress_counter["count"] += len(ok_map)
                    if (progress_counter["count"] // 50) > (prev // 50):
                        _log_gpu_progress(
                            progress_counter["count"], len(work_units), gpu_log, skip_gpu_check
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

    roundtrip_mismatches = 0
    for tw_s in zh_tw_texts:
        # Compare t2s round-trip stability on s2t output
        if tw_mode == "s2t" and to_simplified(tw_s) != to_simplified(to_traditional(to_simplified(tw_s), "s2t")):
            roundtrip_mismatches += 1

    qa_summary = {
        "job": job,
        "total_records": len(records),
        "unique_work_units": len(work_units),
        "passthrough_records": len(passthrough_cn),
        "missing": len(missing_ids),
        "failures": len(all_failures),
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
    ap = argparse.ArgumentParser(description="Colab L4 Sakura translation runner")
    ap.add_argument("--drive-root", default=f"/content/drive/MyDrive/{schema.DRIVE_ROOT_NAME}")
    ap.add_argument("--api-base", help="Use an already-running OpenAI-compatible /v1 endpoint")
    ap.add_argument("--tw-mode", default="s2t", choices=("s2t", "s2twp"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-gpu-check", action="store_true")
    ap.add_argument("--manifest-wait", type=float, default=60.0)
    ap.add_argument("--min-tps", type=float, default=DEFAULT_MIN_TPS)
    ap.add_argument("--rebuild-bin", action="store_true")
    args = ap.parse_args(argv)

    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")

    dirs = schema.ensure_drive_tree(args.drive_root)
    input_dir = dirs["sakura_input"]
    work_dir = dirs["sakura_work"]
    output_dir = dirs["sakura_output"]

    # Strict check: reject non-.jsonl/.manifest.json files inside input_dir
    for p in sorted(input_dir.iterdir()):
        if not p.is_file():
            continue
        if p.name.endswith(".manifest.json"):
            continue
        if p.suffix.lower() != ".jsonl":
            raise schema.SchemaError(
                f"Unexpected non-.jsonl file in {input_dir}: {p.name}. Remove it before running."
            )

    job_files = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jsonl")
    if not job_files:
        print(f"[runner] No .jsonl jobs found in {input_dir}")
        return 0

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
            local_bin = Path("/content/bin")
            local_models = Path("/content/models")
            llama_log = Path("/content/llama.log")
            llama_exe = ensure_llama_server(
                dirs["root"], local_bin, force_rebuild=args.rebuild_bin
            )
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
                shutil.copy2(local_log, work_dir / "llama.log")
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
