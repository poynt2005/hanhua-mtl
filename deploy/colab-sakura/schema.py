"""Shared strict schema, Drive layout, manifest, and DONE-marker helpers.

100% Python stdlib only (no third-party imports) so local Windows agents can
run validation and Drive transfers with zero dependency setup.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DRIVE_ROOT_NAME = "galgame_mtl"
SUBDIR = "sakura"

ALLOWED_KINDS = ("dialogue", "narration", "choice", "ui", "system")
ALLOWED_RECORD_KEYS = frozenset({"id", "text", "kind", "file", "speaker", "override"})
MAX_TEXT_LEN = 1500
JOB_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
PLACEHOLDER_RE = re.compile(r"\{\{[^}]+\}\}|⟦\d+⟧|\\[A-Za-z]+\[\d+\]")

# Inline engine control codes such as "_w!10,0,". The runner keeps these away from
# the model: text is split around them and they are re-inserted byte-for-byte.
CONTROL_CODE_RE = re.compile(r"_[A-Za-z]+!(?:-?\d+,)+")
# Other markup shapes that are NOT protected automatically (used for warnings only).
SUSPECT_CODE_RE = re.compile(r"<[A-Za-z/][^<>]{0,40}>|\\[A-Za-z]+|%[a-z]|\[[a-z]+[=\]]|_[A-Za-z]+!")

WATCH_CHAR_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("发", ("发", "髮", "發")),
    ("后", ("后", "後")),
    ("里", ("里", "裡")),
    ("面", ("面", "麵")),
    ("干", ("干", "乾", "幹")),
    ("只", ("只", "隻")),
    ("系", ("系", "係", "繫")),
    ("制", ("制", "製")),
    ("准", ("准", "準")),
    ("复", ("复", "復", "複")),
    ("历", ("历", "歷", "曆")),
    ("尽", ("尽", "盡", "儘")),
    ("获", ("获", "獲", "穫")),
    ("钟", ("钟", "鐘", "鍾")),
)

MODELS: dict[str, dict[str, Any]] = {
    "sakura": {
        "key": "sakura",
        "alias": "sakura-galtransl-14b",
        "repo": "SakuraLLM/Sakura-GalTransl-14B-v3.8",
        "rev": "10bab61b45d8970dc5c5fd7a4aec920d49a3cd8e",
        "filename": "Sakura-Galtransl-14B-v3.8.gguf",
        "size_bytes": 12124681536,
        "sha256": "6e32fcf5bfdf78114745d2bf6bd87fb107ad7f68a611f4c2a886a4378846e334",
        "url": (
            "https://huggingface.co/SakuraLLM/Sakura-GalTransl-14B-v3.8/resolve/"
            "10bab61b45d8970dc5c5fd7a4aec920d49a3cd8e/Sakura-Galtransl-14B-v3.8.gguf"
        ),
        "mirror_url": (
            "https://hf-mirror.com/SakuraLLM/Sakura-GalTransl-14B-v3.8/resolve/"
            "10bab61b45d8970dc5c5fd7a4aec920d49a3cd8e/Sakura-Galtransl-14B-v3.8.gguf"
        ),
    },
    "hymt2": {
        "key": "hymt2",
        "alias": "hy-mt2-heretic",
        "repo": "mradermacher/Hy-MT2-30B-A3B-uncensored-heretic-i1-GGUF",
        "rev": "862b7d66847b339c0594b8de39d38b035ee75f25",
        "filename": "Hy-MT2-30B-A3B-uncensored-heretic.i1-Q4_K_M.gguf",
        "size_bytes": 18236703488,
        "sha256": "b0358d0319aae3f821fef4989bd85f409336139bfc7db59fd049848b9882321c",
        "url": (
            "https://huggingface.co/mradermacher/Hy-MT2-30B-A3B-uncensored-heretic-i1-GGUF/resolve/"
            "862b7d66847b339c0594b8de39d38b035ee75f25/"
            "Hy-MT2-30B-A3B-uncensored-heretic.i1-Q4_K_M.gguf"
        ),
        "mirror_url": (
            "https://hf-mirror.com/mradermacher/Hy-MT2-30B-A3B-uncensored-heretic-i1-GGUF/resolve/"
            "862b7d66847b339c0594b8de39d38b035ee75f25/"
            "Hy-MT2-30B-A3B-uncensored-heretic.i1-Q4_K_M.gguf"
        ),
    },
}


class SchemaError(ValueError):
    """Raised when an input JSONL, glossary, manifest, or DONE file violates constraints."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str | Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: str | Path, content: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return target


def atomic_write_json(path: str | Path, obj: Any) -> Path:
    return atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def ensure_drive_tree(drive_root: str | Path) -> dict[str, Path]:
    root = Path(drive_root)
    dirs = {
        "root": root,
        "bin": root / "bin",
        "models": root / "models",
        "scripts_sakura": root / "scripts" / "sakura",
        "scripts_hymt2": root / "scripts" / "hymt2",
        "sakura_input": root / "sakura" / "input",
        "sakura_work": root / "sakura" / "work",
        "sakura_output": root / "sakura" / "output",
        "hymt2_input": root / "hymt2" / "input",
        "hymt2_work": root / "hymt2" / "work",
        "hymt2_output": root / "hymt2" / "output",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def validate_job_name(job: str) -> str:
    if not job or not JOB_NAME_RE.match(job):
        raise SchemaError(f"Invalid job name {job!r}: must match ^[A-Za-z0-9._-]+$")
    if job.endswith(".zh_CN") or job.endswith(".zh_TW"):
        raise SchemaError(f"Invalid job name {job!r}: must not end with .zh_CN or .zh_TW")
    return job


def extract_placeholders(text: str) -> list[str]:
    return PLACEHOLDER_RE.findall(text or "")


def extract_control_codes(text: str) -> list[str]:
    return CONTROL_CODE_RE.findall(text or "")


def find_unprotected_codes(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count markup-like tokens the runner will send to the model as plain text."""
    found: dict[str, int] = {}
    for rec in records:
        rest = PLACEHOLDER_RE.sub("", CONTROL_CODE_RE.sub("", rec.get("text") or ""))
        for token in SUSPECT_CODE_RE.findall(rest):
            found[token] = found.get(token, 0) + 1
    return found


def count_watch_chars(zh_cn_texts: list[str], zh_tw_texts: list[str]) -> dict[str, dict[str, int]]:
    report: dict[str, dict[str, int]] = {}
    joined_cn = "\n".join(zh_cn_texts)
    joined_tw = "\n".join(zh_tw_texts)
    for key, variants in WATCH_CHAR_GROUPS:
        cn_count = joined_cn.count(key)
        tw_counts = {v: joined_tw.count(v) for v in variants}
        if cn_count or any(tw_counts.values()):
            report[key] = {"zh_CN": cn_count, **{f"tw_{k}": v for k, v in tw_counts.items()}}
    return report


def validate_jsonl_file(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        raise SchemaError(f"Input file not found: {p}")
    if p.suffix.lower() != ".jsonl":
        raise SchemaError(f"{p.name}: input file must have .jsonl extension")
    validate_job_name(p.stem)

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with open(p, "r", encoding="utf-8-sig") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                raise SchemaError(f"{p.name}:{lineno}: blank lines are not allowed in strict JSONL")
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"{p.name}:{lineno}: invalid JSON ({exc})") from exc
            if not isinstance(obj, dict):
                raise SchemaError(f"{p.name}:{lineno}: line must be a JSON object")

            unknown = set(obj.keys()) - ALLOWED_RECORD_KEYS
            if unknown:
                raise SchemaError(f"{p.name}:{lineno}: unknown keys {sorted(unknown)}")

            rec_id = obj.get("id")
            if not isinstance(rec_id, str) or not rec_id.strip():
                raise SchemaError(f"{p.name}:{lineno}: 'id' must be a non-empty string")
            if rec_id in seen_ids:
                raise SchemaError(f"{p.name}:{lineno}: duplicate id {rec_id!r}")
            seen_ids.add(rec_id)

            text = obj.get("text")
            if not isinstance(text, str):
                raise SchemaError(f"{p.name}:{lineno}: 'text' must be a string")
            if "\x00" in text:
                raise SchemaError(f"{p.name}:{lineno}: 'text' contains NUL byte")
            if len(text) > MAX_TEXT_LEN:
                raise SchemaError(
                    f"{p.name}:{lineno}: 'text' length {len(text)} exceeds {MAX_TEXT_LEN}"
                )

            kind = obj.get("kind")
            if not isinstance(kind, str) or kind not in ALLOWED_KINDS:
                raise SchemaError(
                    f"{p.name}:{lineno}: 'kind' must be one of {ALLOWED_KINDS}, got {kind!r}"
                )

            file_val = obj.get("file", "default")
            if not isinstance(file_val, str) or not file_val.strip():
                raise SchemaError(f"{p.name}:{lineno}: 'file' must be a non-empty string")

            speaker = obj.get("speaker", None)
            if speaker is not None and not isinstance(speaker, str):
                raise SchemaError(f"{p.name}:{lineno}: 'speaker' must be a string or null")

            override = obj.get("override", None)
            if override is not None and not isinstance(override, str):
                raise SchemaError(f"{p.name}:{lineno}: 'override' must be a string or null")

            records.append(
                {
                    "id": rec_id,
                    "file": file_val,
                    "kind": kind,
                    "speaker": speaker,
                    "text": text,
                    "override": override,
                }
            )

    if not records:
        raise SchemaError(f"{p.name}: input file contains 0 records")
    return records


def validate_glossary_file(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.is_file():
        raise SchemaError(f"Glossary file not found: {p}")
    try:
        with open(p, "r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{p.name}: invalid JSON ({exc})") from exc

    entries: list[dict[str, str]] = []
    if isinstance(data, dict):
        for src, dst in data.items():
            if not isinstance(src, str) or not src.strip() or not isinstance(dst, str) or not dst.strip():
                raise SchemaError(f"{p.name}: dict glossary entries must be non-empty str -> str")
            entries.append({"src": src, "dst": dst, "note": ""})
    elif isinstance(data, list):
        for idx, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                raise SchemaError(f"{p.name}[{idx}]: list glossary item must be an object")
            extra = set(item.keys()) - {"src", "dst", "note"}
            if extra:
                raise SchemaError(f"{p.name}[{idx}]: unknown keys {sorted(extra)}")
            src = item.get("src")
            dst = item.get("dst")
            note = item.get("note", "")
            if not isinstance(src, str) or not src.strip():
                raise SchemaError(f"{p.name}[{idx}]: 'src' must be a non-empty string")
            if not isinstance(dst, str) or not dst.strip():
                raise SchemaError(f"{p.name}[{idx}]: 'dst' must be a non-empty string")
            if not isinstance(note, str):
                raise SchemaError(f"{p.name}[{idx}]: 'note' must be a string")
            entries.append({"src": src, "dst": dst, "note": note})
    else:
        raise SchemaError(f"{p.name}: glossary must be a JSON object or array")
    return entries


def manifest_path_for(input_path: str | Path) -> Path:
    p = Path(input_path)
    return p.with_name(f"{p.stem}.manifest.json")


def build_manifest(
    job: str,
    input_path: str | Path,
    record_count: int,
    glossary_path: str | Path | None = None,
) -> dict[str, Any]:
    p = Path(input_path)
    g_sha = sha256_file(glossary_path) if glossary_path and Path(glossary_path).is_file() else None
    return {
        "schema_version": 1,
        "job": job,
        "filename": p.name,
        "size_bytes": p.stat().st_size,
        "sha256": sha256_file(p),
        "record_count": record_count,
        "glossary_sha256": g_sha,
        "created_at_utc": utc_now_iso(),
    }


def verify_manifest(
    input_path: str | Path,
    manifest_path: str | Path | None = None,
    wait_sec: float = 0.0,
    poll_sec: float = 5.0,
) -> dict[str, Any]:
    inp = Path(input_path)
    mpath = Path(manifest_path) if manifest_path else manifest_path_for(inp)
    deadline = time.monotonic() + max(0.0, float(wait_sec))

    last_err = "manifest missing"
    while True:
        if inp.is_file() and mpath.is_file():
            try:
                with open(mpath, "r", encoding="utf-8-sig") as handle:
                    manifest = json.load(handle)
                exp_size = int(manifest["size_bytes"])
                exp_sha = str(manifest["sha256"])
                act_size = inp.stat().st_size
                if act_size != exp_size:
                    last_err = f"{inp.name}: size {act_size} != manifest {exp_size}"
                else:
                    act_sha = sha256_file(inp)
                    if act_sha != exp_sha:
                        last_err = f"{inp.name}: sha256 {act_sha[:12]} != manifest {exp_sha[:12]}"
                    else:
                        return manifest
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                last_err = f"{mpath.name}: unreadable ({exc})"
        else:
            last_err = f"waiting for {inp.name} and {mpath.name}"

        if time.monotonic() >= deadline:
            raise SchemaError(f"Manifest verification failed: {last_err}")
        time.sleep(min(poll_sec, max(0.1, deadline - time.monotonic())))


def done_marker_path(output_dir: str | Path, job: str) -> Path:
    return Path(output_dir) / f"{job}.DONE.json"


def build_done_marker(
    job: str,
    output_files: list[Path],
    qa_summary: dict[str, Any],
) -> dict[str, Any]:
    files_meta: dict[str, dict[str, Any]] = {}
    for fp in output_files:
        files_meta[fp.name] = {
            "size_bytes": fp.stat().st_size,
            "sha256": sha256_file(fp),
        }
    return {
        "schema_version": 1,
        "job": job,
        "completed_at_utc": utc_now_iso(),
        "files": files_meta,
        "qa_summary": qa_summary,
    }


def verify_done_marker(output_dir: str | Path, job: str) -> dict[str, Any]:
    dpath = done_marker_path(output_dir, job)
    if not dpath.is_file():
        raise SchemaError(f"DONE marker missing: {dpath}")
    try:
        with open(dpath, "r", encoding="utf-8-sig") as handle:
            marker = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaError(f"DONE marker unreadable ({dpath}): {exc}") from exc

    files = marker.get("files")
    if not isinstance(files, dict) or not files:
        raise SchemaError(f"DONE marker {dpath.name} lists no files")

    for fname, meta in files.items():
        fp = Path(output_dir) / fname
        if not fp.is_file():
            raise SchemaError(f"Output file missing for DONE marker: {fp}")
        if fp.stat().st_size != int(meta["size_bytes"]):
            raise SchemaError(f"Output size mismatch for {fname}")
        if sha256_file(fp) != str(meta["sha256"]):
            raise SchemaError(f"Output SHA-256 mismatch for {fname}")
    return marker
