"""Option A local Google Drive bridge for hymt2-heretic-colab (100% Python stdlib).

Auto-detects the mounted Google Drive containing `galgame_mtl` (works across
multi-account Drive Desktop setups such as G:\\ and H:\\), enforces strict
JSONL schema validation before pushing, writes atomic `.manifest.json` SHA-256
manifests, syncs runner scripts, checks Colab status, and pulls verified outputs.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import string
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import schema  # noqa: E402

DRIVE_SUBFOLDERS = ("我的雲端硬碟", "My Drive")
SYNC_FILES = ("schema.py", "runner.py")


def is_drive_desktop_running() -> bool:
    if os.name != "nt":
        return True
    res = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq GoogleDriveFS.exe"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return "GoogleDriveFS.exe" in res.stdout


def discover_drive_root(explicit: str | None = None) -> Path:
    env_val = explicit or os.environ.get("GALGAME_MTL_DRIVE_ROOT")
    if env_val:
        p = Path(env_val)
        if not p.exists():
            raise RuntimeError(f"Configured Drive root does not exist: {p}")
        return p

    candidates: list[Path] = []
    for letter in string.ascii_uppercase:
        if letter in ("A", "B", "C"):
            continue
        drive = Path(f"{letter}:\\")
        if not drive.exists():
            continue
        for sub in DRIVE_SUBFOLDERS:
            cand = drive / sub / schema.DRIVE_ROOT_NAME
            if cand.is_dir():
                candidates.append(cand)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise RuntimeError(
            f"No mounted Google Drive containing '{schema.DRIVE_ROOT_NAME}' was found. "
            "Run deploy_models_colab.ipynb on Colab first, or pass --drive-root / set GALGAME_MTL_DRIVE_ROOT."
        )
    joined = ", ".join(str(c) for c in candidates)
    raise RuntimeError(
        f"Multiple Google Drives contain '{schema.DRIVE_ROOT_NAME}': {joined}. "
        "Disambiguate with --drive-root or GALGAME_MTL_DRIVE_ROOT."
    )


def _require_drivefs(skip_check: bool) -> None:
    if not skip_check and not is_drive_desktop_running():
        raise RuntimeError(
            "GoogleDriveFS.exe is not running. Start Google Drive for Desktop so pushed files sync to the cloud."
        )


def _sync_scripts(dirs: dict[str, Path]) -> None:
    dest_dir = dirs["scripts_hymt2"]
    for name in SYNC_FILES:
        src = SCRIPT_DIR / name
        if src.is_file():
            dst = dest_dir / name
            tmp = dst.with_name(f".{name}.tmp.{os.getpid()}")
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)


def cmd_doctor(args: argparse.Namespace) -> int:
    _require_drivefs(args.skip_drivefs_check)
    root = discover_drive_root(args.drive_root)
    dirs = schema.ensure_drive_tree(root)
    spec = schema.MODELS["hymt2"]
    model_path = dirs["models"] / spec["filename"]

    print("[doctor] DriveFS running : OK")
    print(f"[doctor] Drive root      : {root}")
    if not model_path.is_file():
        print(f"[doctor] Model           : MISSING ({model_path})")
        print("         Action: open Free CPU Colab and run deploy_models_colab.ipynb")
        return 1

    sz = model_path.stat().st_size
    size_ok = sz == spec["size_bytes"]
    print(f"[doctor] Model size      : {sz} / {spec['size_bytes']} ({'OK' if size_ok else 'MISMATCH'})")
    if not size_ok:
        return 1

    if args.verify_hash:
        h = schema.sha256_file(model_path)
        hash_ok = h == spec["sha256"]
        print(f"[doctor] Model SHA-256   : {h} ({'OK' if hash_ok else 'MISMATCH'})")
        if not hash_ok:
            return 1

    bin_tar = dirs["bin"] / "llama-server-l4-b12019.tar.gz"
    if bin_tar.is_file():
        print(f"[doctor] Prebuilt bin    : OK ({bin_tar.stat().st_size} bytes)")
    else:
        print("[doctor] Prebuilt bin    : MISSING (will auto-download from GitHub Release on Colab)")

    inputs = sorted(p.name for p in dirs["hymt2_input"].glob("*.jsonl"))
    dones = sorted(p.name for p in dirs["hymt2_output"].glob("*.DONE.json"))
    print(f"[doctor] Pending inputs  : {inputs}")
    print(f"[doctor] Completed DONEs : {dones}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    records = schema.validate_jsonl_file(args.input)
    g_count = 0
    if args.glossary:
        g_count = len(schema.validate_glossary_file(args.glossary))
    print(f"[validate] PASS: {Path(args.input).name} ({len(records)} records, {g_count} glossary terms)")
    return 0


def cmd_push(args: argparse.Namespace) -> int:
    _require_drivefs(args.skip_drivefs_check)
    root = discover_drive_root(args.drive_root)
    dirs = schema.ensure_drive_tree(root)

    src_input = Path(args.input)
    records = schema.validate_jsonl_file(src_input)
    job = schema.validate_job_name(args.name or src_input.stem)

    glossary_entries = None
    if args.glossary:
        glossary_entries = schema.validate_glossary_file(args.glossary)

    dst_input = dirs["hymt2_input"] / f"{job}.jsonl"
    if dst_input.exists() and not args.force:
        raise RuntimeError(f"Job already exists on Drive: {dst_input}. Pass --force to overwrite.")

    old_done = schema.done_marker_path(dirs["hymt2_output"], job)
    if args.force and old_done.exists():
        old_done.unlink()

    _sync_scripts(dirs)

    dst_glossary: Path | None = None
    existing_glossary = dirs["root"] / schema.SUBDIR / "glossary.json"
    if args.glossary and glossary_entries is not None:
        dst_glossary = existing_glossary
        schema.atomic_write_json(dst_glossary, glossary_entries)
    elif existing_glossary.is_file():
        print(f"[push] NOTE: No --glossary given, but existing {existing_glossary} is present on Drive.")

    src_sha = schema.sha256_file(src_input)
    tmp_input = dst_input.with_name(f".{dst_input.name}.tmp.{os.getpid()}")
    shutil.copyfile(src_input, tmp_input)
    if schema.sha256_file(tmp_input) != src_sha:
        tmp_input.unlink(missing_ok=True)
        raise RuntimeError("SHA-256 mismatch after copying input to Drive staging file")
    os.replace(tmp_input, dst_input)

    manifest = schema.build_manifest(job, dst_input, len(records), dst_glossary)
    mpath = schema.manifest_path_for(dst_input)
    schema.atomic_write_json(mpath, manifest)

    print(f"[push] OK: {dst_input}")
    print(f"[push] Manifest: {mpath.name} (records={len(records)}, sha256={manifest['sha256'][:16]}...)")
    print("[push] Next step for operator:")
    print("       Open hymt2_colab.ipynb in Google Colab (L4 GPU) and press Ctrl+F9 (Run all).")
    return 0


def _job_status(dirs: dict[str, Path], job: str) -> dict[str, Any]:
    inp = dirs["hymt2_input"] / f"{job}.jsonl"
    ckpt = dirs["hymt2_work"] / f"{job}.checkpoint.jsonl"
    fail = dirs["hymt2_work"] / f"{job}.failures.jsonl"
    qa_path = dirs["hymt2_output"] / f"{job}.qa_report.json"

    ckpt_ids: set[str] = set()
    if ckpt.is_file():
        with open(ckpt, "r", encoding="utf-8-sig") as fh:
            for ln in fh:
                s = ln.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                    if obj.get("wid") and str(obj.get("zh_cn", "")).strip():
                        ckpt_ids.add(obj["wid"])
                except ValueError:
                    pass

    fail_count = 0
    if fail.is_file():
        with open(fail, "r", encoding="utf-8-sig") as fh:
            fail_count = sum(1 for ln in fh if ln.strip())

    done_ok = False
    done_err = None
    if schema.done_marker_path(dirs["hymt2_output"], job).is_file():
        try:
            schema.verify_done_marker(dirs["hymt2_output"], job)
            done_ok = True
        except schema.SchemaError as exc:
            done_err = str(exc)

    qa = None
    if qa_path.is_file():
        with open(qa_path, "r", encoding="utf-8-sig") as fh:
            qa = json.load(fh)

    return {
        "job": job,
        "input_present": inp.is_file(),
        "checkpoint_units": len(ckpt_ids),
        "failures": fail_count,
        "done_verified": done_ok,
        "done_error": done_err,
        "qa": qa,
    }


def cmd_status(args: argparse.Namespace) -> int:
    _require_drivefs(args.skip_drivefs_check)
    root = discover_drive_root(args.drive_root)
    dirs = schema.ensure_drive_tree(root)

    jobs = (
        [schema.validate_job_name(args.name)]
        if args.name
        else sorted({p.stem for p in dirs["hymt2_input"].glob("*.jsonl")})
    )
    if not jobs:
        print("[status] No jobs found")
        return 0

    for j in jobs:
        st = _job_status(dirs, j)
        print(json.dumps(st, ensure_ascii=False))
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    _require_drivefs(args.skip_drivefs_check)
    root = discover_drive_root(args.drive_root)
    dirs = schema.ensure_drive_tree(root)
    job = schema.validate_job_name(args.name)

    inp_path = dirs["hymt2_input"] / f"{job}.jsonl"
    if not inp_path.is_file():
        raise RuntimeError(f"Original Drive input missing: {inp_path}")
    in_records = schema.validate_jsonl_file(inp_path)
    in_ids = [r["id"] for r in in_records]

    marker = schema.verify_done_marker(dirs["hymt2_output"], job)
    tw_out = dirs["hymt2_output"] / f"{job}.zh_TW.jsonl"
    cn_out = dirs["hymt2_output"] / f"{job}.zh_CN.jsonl"
    qa_out = dirs["hymt2_output"] / f"{job}.qa_report.json"

    for out_file in (cn_out, tw_out):
        out_ids: list[str] = []
        with open(out_file, "r", encoding="utf-8-sig") as fh:
            for ln in fh:
                if ln.strip():
                    obj = json.loads(ln)
                    out_ids.append(obj["id"])
        if out_ids != in_ids:
            raise RuntimeError(
                f"ID/order parity mismatch between {inp_path.name} and {out_file.name}"
            )

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for src in (tw_out, cn_out, qa_out, schema.done_marker_path(dirs["hymt2_output"], job)):
        if src.is_file():
            dst = dest / src.name
            tmp = dst.with_name(f".{src.name}.tmp.{os.getpid()}")
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)

    print(
        f"[pull] OK: {job} ({len(in_ids)} records, verified SHA-256) -> {dest}"
    )
    print(f"[pull] QA summary: {json.dumps(marker.get('qa_summary', {}), ensure_ascii=False)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Option A local Drive bridge for hymt2-heretic-colab")
    ap.add_argument("--drive-root", help="Explicit path to galgame_mtl on mounted Google Drive")
    ap.add_argument("--skip-drivefs-check", action="store_true", help="Skip GoogleDriveFS.exe check (for tests)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="Verify Drive mount, model file, and directories")
    d.add_argument("--verify-hash", action="store_true", help="Also compute full SHA-256 of local Drive GGUF")

    v = sub.add_parser("validate", help="Validate strict JSONL input and optional glossary")
    v.add_argument("--input", required=True)
    v.add_argument("--glossary")

    p = sub.add_parser("push", help="Validate and atomically push job + manifest + runner scripts to Drive")
    p.add_argument("--input", required=True)
    p.add_argument("--glossary")
    p.add_argument("--name")
    p.add_argument("--force", action="store_true")

    s = sub.add_parser("status", help="Inspect Drive job progress, QA report, and DONE marker")
    s.add_argument("--name")

    pl = sub.add_parser("pull", help="Verify DONE marker & SHA-256 and pull outputs locally")
    pl.add_argument("--name", required=True)
    pl.add_argument("--dest", required=True)

    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    dispatch = {
        "doctor": cmd_doctor,
        "validate": cmd_validate,
        "push": cmd_push,
        "status": cmd_status,
        "pull": cmd_pull,
    }
    return dispatch[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
