"""Comprehensive offline unit + integration tests for deploy/colab-sakura.

Runs completely locally in temporary directories using a mock SSE OpenAI server.
Leaves zero artifacts in the repository.
"""
from __future__ import annotations

import ast
import contextlib
import http.server
import io
import json
import os
import re
import socketserver
import sys
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import drive_bridge  # noqa: E402
import runner  # noqa: E402
import schema  # noqa: E402


def _mock_translate_line(src: str) -> str:
    if "おはよう" in src:
        return "早安。睡得好吗？"
    if "差し込んだ" in src:
        return "\u3000清晨静谧的阳光洒进了房间。"
    if "敏感なんだから" in src:
        return "啊……不行，那里很敏感的……♥"
    if "「今日は" in src and "」" not in src:
        return "「今天也是"
    if "ありがとう」" in src and "「" not in src:
        return "谢谢你」"
    if "ユイ：" in src:
        # Simulate mirrored speaker prefix + Traditional char leak (測試 -> 繁體) + ♥ + placeholder
        body = src.split("：", 1)[1]
        out = body.replace("テスト", "测试").replace("元気", "元氣")
        out = re.sub(r"[\u3040-\u309f\u30a0-\u30ff]+", "", out)
        return f"由衣：「{out}" if "「" in body else f"由衣：{out}"
    if "髪" in src:
        return "头发后来很长。"
    cleaned = re.sub(r"[\u3040-\u309f\u30a0-\u30ff]+", "的", src)
    return cleaned.replace("テスト", "测试").replace("セーブ", "存档").replace("設定", "设置")


class _MockState:
    def __init__(self, fail_batch_once: bool = True):
        self.lock = threading.Lock()
        self.fail_batch_once = fail_batch_once
        self.did_fail_batch = False
        self.ui_saw_history = False
        self.calls = 0


def _make_handler(state: _MockState):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            if self.path == "/health":
                body = json.dumps({"status": "ok"}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/v1/models":
                body = json.dumps({"data": [{"id": runner.MODEL_ALIAS}]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            req_obj = json.loads(self.rfile.read(length).decode("utf-8"))
            user_msg = req_obj["messages"][-1]["content"]

            # Detect history presence for UI lines
            hist_match = re.search(r"历史翻译：\n(.*?)\n\n参考以下术语表", user_msg, re.S)
            hist_content = hist_match.group(1).strip() if hist_match else ""
            marker = "将下面的文本从日文翻译成简体中文："
            idx = user_msg.find(marker)
            if idx == -1:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing Sakura prompt marker")
                return
            after = user_msg[idx + len(marker):]
            if not after.startswith("\n"):
                self.send_response(400)
                self.end_headers()
                self.wfile.write(f"Unexpected text after prompt marker: {after[:80]}".encode("utf-8"))
                return
            src_block = after[1:]
            src_lines = [ln for ln in src_block.splitlines() if ln.strip()]

            with state.lock:
                state.calls += 1
                if any("設定" in ln for ln in src_lines) and hist_content:
                    state.ui_saw_history = True

                # Trigger line-count mismatch once on a multi-line batch to test 8->4->2->1 split
                trigger_bad_count = False
                if state.fail_batch_once and not state.did_fail_batch and len(src_lines) >= 2:
                    state.did_fail_batch = True
                    trigger_bad_count = True

            if trigger_bad_count:
                reply_lines = ["只有一行"]
            else:
                reply_lines = [_mock_translate_line(ln) for ln in src_lines]
            reply = "\n".join(reply_lines)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            # Emit SSE chunks
            for ch in reply:
                ev = json.dumps({"choices": [{"delta": {"content": ch}}]}, ensure_ascii=False)
                self.wfile.write(f"data: {ev}\n\n".encode("utf-8"))
            end_ev = json.dumps(
                {"choices": [{"delta": {}}], "usage": {"completion_tokens": max(20, len(reply) * 2)}}
            )
            self.wfile.write(f"data: {end_ev}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    return Handler


class _ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


class SakuraColabTests(unittest.TestCase):
    def test_schema_validation_and_bom(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "demo.jsonl"
            # UTF-8 with BOM should succeed
            row = json.dumps({"id": "1", "kind": "dialogue", "speaker": "A", "text": "テスト"}, ensure_ascii=False)
            p.write_bytes(b"\xef\xbb\xbf" + (row + "\n").encode("utf-8"))
            recs = schema.validate_jsonl_file(p)
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0]["file"], "default")

            # Unknown key
            bad1 = Path(td) / "bad1.jsonl"
            bad1.write_text(json.dumps({"id": "1", "kind": "ui", "text": "x", "extra": 1}) + "\n", encoding="utf-8")
            with self.assertRaises(schema.SchemaError):
                schema.validate_jsonl_file(bad1)

            # Duplicate ID
            bad2 = Path(td) / "bad2.jsonl"
            line = json.dumps({"id": "1", "kind": "ui", "text": "x"})
            bad2.write_text(f"{line}\n{line}\n", encoding="utf-8")
            with self.assertRaises(schema.SchemaError):
                schema.validate_jsonl_file(bad2)

            # Invalid kind
            bad3 = Path(td) / "bad3.jsonl"
            bad3.write_text(json.dumps({"id": "1", "kind": "invalid", "text": "x"}) + "\n", encoding="utf-8")
            with self.assertRaises(schema.SchemaError):
                schema.validate_jsonl_file(bad3)

            # >1500 chars
            bad4 = Path(td) / "bad4.jsonl"
            bad4.write_text(json.dumps({"id": "1", "kind": "ui", "text": "a" * 1501}) + "\n", encoding="utf-8")
            with self.assertRaises(schema.SchemaError):
                schema.validate_jsonl_file(bad4)

            # NUL byte
            bad5 = Path(td) / "bad5.jsonl"
            bad5.write_text(json.dumps({"id": "1", "kind": "ui", "text": "a\x00b"}) + "\n", encoding="utf-8")
            with self.assertRaises(schema.SchemaError):
                schema.validate_jsonl_file(bad5)

            # Glossary list & dict
            g1 = Path(td) / "g1.json"
            g1.write_text(json.dumps({"ユイ": "由衣"}), encoding="utf-8")
            self.assertEqual(schema.validate_glossary_file(g1)[0]["dst"], "由衣")
            g2 = Path(td) / "g2.json"
            g2.write_text(json.dumps([{"src": "ユイ", "dst": "由衣", "note": "女"}]), encoding="utf-8")
            self.assertEqual(schema.validate_glossary_file(g2)[0]["note"], "女")

    def test_normalization_and_validation_rules(self):
        # Longest-name-first speaker prefix stripping
        u = {"kind": "dialogue", "speaker": "男人", "text": "走吧♥{{TAG_0}}"}
        cn, raw, flags = runner.normalize_output_line("男人：走吧♥{{TAG_0}}", u, ["男人", "男"])
        self.assertEqual(cn, "走吧♥{{TAG_0}}")
        self.assertEqual(runner.validate_unit_translation(u["text"], cn), [])

        # Inner speaker prefix + duplicate closing quote + Traditional t2s normalization
        u2 = {"kind": "dialogue", "speaker": "不良A", "text": "「大丈夫？」"}
        cn2, raw2, flags2 = runner.normalize_output_line("「不良少年A：「沒問題？」」", u2, ["不良A"])
        self.assertEqual(cn2, "「没问题？」")
        self.assertEqual(raw2, "「沒問題？」")
        self.assertIn("t2s", flags2)
        self.assertEqual(runner.validate_unit_translation(u2["text"], cn2), [])

        # Asymmetric split quotes must match source, and auto-closed split quotes are normalized
        self.assertEqual(runner.validate_unit_translation("「今日は", "「今天也是"), [])
        self.assertIn("quote_asymmetry_mismatch", runner.validate_unit_translation("「今日は", "「今天也是」"))
        u_open = {"kind": "dialogue", "speaker": "ユイ", "text": "「今日は"}
        cn_open, _, flags_open = runner.normalize_output_line("由衣：「今天天气真好」", u_open, ["ユイ"])
        self.assertEqual(cn_open, "「今天天气真好")
        self.assertIn("trim_close_quote", flags_open)
        self.assertEqual(runner.validate_unit_translation(u_open["text"], cn_open), [])
        u_close = {"kind": "dialogue", "speaker": "ユイ", "text": "ありがとう」"}
        cn_close, _, flags_close = runner.normalize_output_line("由衣：「谢谢你」", u_close, ["ユイ"])
        self.assertEqual(cn_close, "谢谢你」")
        self.assertIn("trim_open_quote", flags_close)
        self.assertEqual(runner.validate_unit_translation(u_close["text"], cn_close), [])

        # Heart glyph swap normalization (♥ <-> ♡) when source uses one exclusively
        u_heart = {"kind": "dialogue", "speaker": "ユイ", "text": "あっ……♥"}
        cn_heart, _, _ = runner.normalize_output_line("啊……♡", u_heart, ["ユイ"])
        self.assertEqual(cn_heart, "啊……♥")
        self.assertEqual(runner.validate_unit_translation(u_heart["text"], cn_heart), [])

        # Official GalTransl v3.8 prompt must have nothing injected between ： and \n
        msgs = runner.build_sakura_messages([u_open], [], [])
        user_content = msgs[-1]["content"]
        self.assertNotIn("请严格输出", user_content)
        marker = "将下面的文本从日文翻译成简体中文："
        self.assertTrue(user_content.split(marker, 1)[1].startswith("\n"))

        # Leading full-width indent restored
        u3 = {"kind": "narration", "speaker": None, "text": "\u3000静か。"}
        cn3, _, _ = runner.normalize_output_line("安静。", u3, [])
        self.assertTrue(cn3.startswith("\u3000"))

        # Kana / placeholder / heart / template leak checks
        self.assertIn("residual_kana", runner.validate_unit_translation("テスト", "测试あ"))
        self.assertIn("placeholder_mismatch", runner.validate_unit_translation("{{TAG_0}}x", "xy"))
        self.assertIn("placeholder_mismatch", runner.validate_unit_translation("{{TAG_0}}{{TAG_1}}", "{{TAG_1}}{{TAG_0}}"))
        self.assertIn("heart_mismatch:♥", runner.validate_unit_translation("好き♥", "喜欢"))
        self.assertIn("template_leak", runner.validate_unit_translation("a", "<|im_start|>hi"))

        # Roundtrip check detects true s2t divergence when t2s(tw) != cn
        self.assertEqual(runner.to_simplified(runner.to_traditional("头发和后来", "s2t")), "头发和后来")
        self.assertNotEqual(runner.to_simplified("頭髮"), "头髮")

    def test_gpu_offload_log_parser(self):
        ok_log = "llm_load_tensors: offloaded 49/49 layers to GPU\nmodel loaded"
        self.assertEqual(runner.parse_gpu_offload_log(ok_log), (49, 49))
        with self.assertRaises(RuntimeError):
            runner.parse_gpu_offload_log("llm_load_tensors: offloaded 20/49 layers to GPU")
        with self.assertRaises(RuntimeError):
            runner.parse_gpu_offload_log("no offload line")
        runner_src = (HERE / "runner.py").read_text(encoding="utf-8")
        self.assertNotIn("LLAMA_ARG_LOG_VERBOSITY", runner_src)
        self.assertIn('"-lv", "4"', runner_src)

    def test_e2e_mock_sse_bridge_resume_and_qa(self):
        state = _MockState(fail_batch_once=True)
        srv = _ThreadedHTTPServer(("127.0.0.1", 0), _make_handler(state))
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            api_base = f"http://127.0.0.1:{port}/v1"
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                drive_root = root / "drive" / schema.DRIVE_ROOT_NAME
                dirs = schema.ensure_drive_tree(drive_root)
                # Put fake model file of exact size for doctor size check
                fake_model = dirs["models"] / schema.MODELS["sakura"]["filename"]
                fake_model.write_bytes(b"")

                local_work = root / "local"
                local_work.mkdir()
                inp_file = local_work / "sample.jsonl"
                gloss_file = local_work / "glossary.json"
                gloss_file.write_text(json.dumps({"ユイ": "由衣", "元気": "精神"}), encoding="utf-8")

                rows = [
                    {"id": "001", "file": "ch01", "kind": "dialogue", "speaker": "ユイ", "text": "おはよう。よく眠れた？"},
                    {"id": "002", "file": "ch01", "kind": "dialogue", "speaker": "ユイ", "text": "テスト♥{{TAG_0}}"},
                    {"id": "003", "file": "ch01", "kind": "dialogue", "speaker": "ユイ", "text": "テスト♥{{TAG_0}}"},  # dup
                    {"id": "004", "file": "ch01", "kind": "dialogue", "speaker": "ユイ", "text": "「今日は"},
                    {"id": "005", "file": "ch01", "kind": "dialogue", "speaker": "ユイ", "text": "ありがとう」"},
                    {"id": "006", "file": "ch01", "kind": "narration", "text": "\u3000静かな朝の日差しが部屋に差し込んだ。"},
                    {"id": "007", "file": "ch01", "kind": "dialogue", "speaker": "ユイ", "text": "……？？"},  # passthrough
                    {"id": "008", "file": "ch01", "kind": "ui", "text": "セーブ", "override": "存檔"},  # override tw->cn
                    {"id": "009", "file": "ch02", "kind": "ui", "text": "設定"},  # ui must get empty history
                    {"id": "010", "file": "ch02", "kind": "narration", "text": "髪が長い。"},  # watch chars
                ]
                inp_file.write_text(
                    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8",
                )

                # Test drive_bridge validate + push + script sync
                rc = drive_bridge.main(
                    ["--drive-root", str(drive_root), "--skip-drivefs-check", "validate",
                     "--input", str(inp_file), "--glossary", str(gloss_file)]
                )
                self.assertEqual(rc, 0)

                rc = drive_bridge.main(
                    ["--drive-root", str(drive_root), "--skip-drivefs-check", "push",
                     "--input", str(inp_file), "--glossary", str(gloss_file), "--name", "game_test"]
                )
                self.assertEqual(rc, 0)
                self.assertTrue((dirs["scripts_sakura"] / "runner.py").is_file())
                self.assertTrue((dirs["sakura_input"] / "game_test.manifest.json").is_file())

                # Seed partial checkpoint to test crash-resume skipping cached unit
                ckpt = dirs["sakura_work"] / "game_test.checkpoint.jsonl"
                ckpt.write_text(
                    json.dumps({"wid": "w000001", "zh_cn": "早安。睡得好吗？"}, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

                # OS metadata files in input/ must be ignored safely
                (dirs["sakura_input"] / "desktop.ini").write_text("[.ShellClassInfo]\n", encoding="utf-8")
                (dirs["sakura_input"] / ".DS_Store").write_bytes(b"\x00")

                rc = runner.main(
                    ["--drive-root", str(drive_root), "--api-base", api_base,
                     "--manifest-wait", "0", "--min-tps", "1.0"]
                )
                self.assertEqual(rc, 0)
                self.assertFalse(state.ui_saw_history, "UI unit leaked story history!")

                # Verify DONE marker & outputs
                marker = schema.verify_done_marker(dirs["sakura_output"], "game_test")
                self.assertEqual(marker["qa_summary"]["missing"], 0)
                self.assertEqual(marker["qa_summary"]["failures"], 0)
                self.assertIn("发", marker["qa_summary"]["watch_chars"])
                self.assertIn("后", marker["qa_summary"]["watch_chars"])
                # 002 & 003 deduplicated, 007 & 008 passthrough => unique_work < total
                self.assertLess(marker["qa_summary"]["unique_work_units"], len(rows))

                # Rerunning without --force should skip cleanly
                calls_before = state.calls
                rc2 = runner.main(
                    ["--drive-root", str(drive_root), "--api-base", api_base,
                     "--manifest-wait", "0", "--skip-gpu-check"]
                )
                self.assertEqual(rc2, 0)
                self.assertEqual(state.calls, calls_before)

                # Test status + pull
                rc = drive_bridge.main(
                    ["--drive-root", str(drive_root), "--skip-drivefs-check", "status", "--name", "game_test"]
                )
                self.assertEqual(rc, 0)

                pulled_dir = local_work / "pulled"
                rc = drive_bridge.main(
                    ["--drive-root", str(drive_root), "--skip-drivefs-check", "pull",
                     "--name", "game_test", "--dest", str(pulled_dir)]
                )
                self.assertEqual(rc, 0)
                tw_lines = [
                    json.loads(ln)
                    for ln in (pulled_dir / "game_test.zh_TW.jsonl").read_text(encoding="utf-8").splitlines()
                    if ln.strip()
                ]
                self.assertEqual([r["id"] for r in tw_lines], [r["id"] for r in rows])
                self.assertEqual(tw_lines[7]["translation"], "存檔")
                self.assertEqual(tw_lines[6]["translation"], "……？？")
        finally:
            srv.shutdown()
            srv.server_close()

    def test_prebuilt_binary_tiers(self):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for name, data in (("llama-server", b"#!/bin/sh\necho ok\n"), ("libggml.so", b"so")):
                ti = tarfile.TarInfo(name=name)
                ti.size = len(data)
                ti.mode = 0o755
                tf.addfile(ti, io.BytesIO(data))
        bundle_bytes = buf.getvalue()
        bundle_sha = runner.hashlib.sha256(bundle_bytes).hexdigest()
        req_count = [0]
        serve_tampered = [False]

        class BinHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def do_GET(self):
                req_count[0] += 1
                if self.path.endswith(".tar.gz"):
                    payload = b"X" * len(bundle_bytes) if serve_tampered[0] else bundle_bytes
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self.send_response(404)
                self.end_headers()

        srv = _ThreadedHTTPServer(("127.0.0.1", 0), BinHandler)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            fake_spec = {
                "filename": "llama-server-l4-b12019.tar.gz",
                "size_bytes": len(bundle_bytes),
                "sha256": bundle_sha,
                "url": f"http://127.0.0.1:{port}/llama-server-l4-b12019.tar.gz",
            }
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                drive_root = root / "drive" / schema.DRIVE_ROOT_NAME
                schema.ensure_drive_tree(drive_root)
                local_bin = root / "local_bin"

                with mock.patch.object(runner, "PREBUILT_BIN_SPEC", fake_spec), \
                     mock.patch.object(runner, "_verify_llama_binary", return_value=True):
                    # 1) Empty Drive bin -> downloads from Release & caches into Drive bin/
                    exe = runner.ensure_llama_server(drive_root, local_bin)
                    self.assertTrue(exe.is_file())
                    self.assertTrue((drive_root / "bin" / fake_spec["filename"]).is_file())
                    self.assertEqual(req_count[0], 1)

                    # 2) Second call -> reuses Drive bin/ with 0 HTTP calls
                    exe2 = runner.ensure_llama_server(drive_root, local_bin)
                    self.assertTrue(exe2.is_file())
                    self.assertEqual(req_count[0], 1)

                    # 3) Tampered archive -> fail fast, no compilation fallback
                    fresh_drive = root / "drive2" / schema.DRIVE_ROOT_NAME
                    schema.ensure_drive_tree(fresh_drive)
                    serve_tampered[0] = True
                    with self.assertRaises(RuntimeError) as ctx:
                        runner.ensure_llama_server(fresh_drive, root / "local_bin2")
                    self.assertIn("sha-256", str(ctx.exception).lower())

            runner_src = (HERE / "runner.py").read_text(encoding="utf-8")
            self.assertNotIn("cmake", runner_src)
            self.assertNotIn("--rebuild-bin", runner_src)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_notebooks_valid_json_and_python_syntax(self):
        for nb_name in ("deploy_models_colab.ipynb", "sakura_colab.ipynb"):
            nb_path = HERE / nb_name
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
            self.assertEqual(nb["nbformat"], 4)
            code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
            self.assertGreaterEqual(len(code_cells), 2)
            full_src = "\n".join("".join(c["source"]) for c in code_cells)
            if nb_name == "deploy_models_colab.ipynb":
                self.assertIn("DEPLOY_LLAMA_BIN", full_src)
                self.assertIn("colab-l4-llama-b12019", full_src)
                self.assertIn(runner.PREBUILT_BIN_SPEC["sha256"], full_src)
            if nb_name == "sakura_colab.ipynb":
                self.assertNotIn("REBUILD_LLAMA_BIN", full_src)
                self.assertNotIn("--rebuild-bin", full_src)
                self.assertIn("DISCONNECT_ON_ERROR = False", full_src)
                self.assertIn("IDLE_UNASSIGN_MINUTES", full_src)
                self.assertIn("REPAIR_ONLY", full_src)
                self.assertIn("STOP THE RUNTIME NOW", full_src)
                self.assertIn("rc == 2", full_src)
                self.assertIn("[PARTIAL]", full_src)
                self.assertIn("rc in (0, 2)", full_src)
                self.assertIn("rc not in (0, 2)", full_src)
            bridge_src = (HERE / "drive_bridge.py").read_text(encoding="utf-8")
            self.assertNotIn("copy2", bridge_src)
            for c in code_cells:
                src = "".join(c["source"])
                cleaned = "\n".join(
                    ln for ln in src.splitlines() if not ln.lstrip().startswith(("!", "%"))
                )
                ast.parse(cleaned, filename=nb_name)
                if nb_name == "sakura_colab.ipynb" and "Popen" in src:
                    self.assertIn("runner.log", src)

    def test_quote_repair_fixtures(self):
        # 1) Asymmetric open-only / close-only split quotes
        q_open, fl_open = runner.repair_quotes("「今日は", "「今天天气真好」")
        self.assertEqual(q_open, "「今天天气真好")
        self.assertIn("trim_close_quote", fl_open)
        self.assertEqual(runner.validate_unit_translation("「今日は", q_open), [])

        q_close, fl_close = runner.repair_quotes("ありがとう」", "「谢谢你」")
        self.assertEqual(q_close, "谢谢你」")
        self.assertIn("trim_open_quote", fl_close)
        self.assertEqual(runner.validate_unit_translation("ありがとう」", q_close), [])

        # 2) Full pair in source, zero in output -> wrap outer pair
        q_pair, fl_pair = runner.repair_quotes("「おはよう」", "早安")
        self.assertEqual(q_pair, "「早安」")
        self.assertIn("add_quote_pair", fl_pair)

        # 3) Duplicated outer quotes -> trim extras
        q_dup, fl_dup = runner.repair_quotes("「大丈夫？」", "「「没问题？」」")
        self.assertEqual(q_dup, "「没问题？」")
        self.assertIn("trim_dup_open_quote", fl_dup)

        # 4) Half-width corner quotes and curly/ASCII quotes converted cleanly
        q_hw, fl_hw = runner.repair_quotes("第1章「勇者の剣」を開始しますか？", "要开始第1章｢勇者之剑｣吗？")
        self.assertEqual(q_hw, "要开始第1章「勇者之剑」吗？")
        self.assertIn("repair_halfwidth_quotes", fl_hw)

        q_curly, fl_curly = runner.repair_quotes("第1章「勇者の剣」を開始しますか？", "要开始第1章“勇者之剑”吗？")
        self.assertEqual(q_curly, "要开始第1章「勇者之剑」吗？")
        self.assertIn("repair_curly_quotes", fl_curly)

        q_title, fl_title = runner.repair_quotes("第1章「勇者の剣」を開始しますか？", "要开始第1章《勇者之剑》吗？")
        self.assertEqual(q_title, "要开始第1章「勇者之剑」吗？")
        self.assertIn("repair_title_quotes", fl_title)

        q_lent, fl_lent = runner.repair_quotes("第1章「勇者の剣」を開始しますか？", "要开始第1章【勇者之剑】吗？")
        self.assertEqual(q_lent, "要开始第1章「勇者之剑」吗？")
        self.assertIn("repair_lenticular_quotes", fl_lent)

        # 5) Inner unfixable quote asymmetry still caught cleanly
        self.assertIn(
            "quote_asymmetry_mismatch",
            runner.validate_unit_translation("小説。「一言」と話す。", "小说。这么说着。"),
        )

    def test_kana_glossary_and_overrides(self):
        # Middle dot ・ (U+30FB) and prolonged dash ー (U+30FC) must NOT false-trigger residual_kana
        self.assertEqual(runner.validate_unit_translation("ＨＰ・ＭＰ回復", "HP・MP回复"), [])
        self.assertEqual(runner.validate_unit_translation("鍵・予備", "钥匙・备用"), [])
        self.assertEqual(runner.validate_unit_translation("うー", "呜ー"), [])
        self.assertFalse(runner.needs_translation({"kind": "ui", "text": "HP・MP"}))
        for bad_ch in ("あ", "ア", "っ", "ッ"):
            self.assertIn("residual_kana", runner.validate_unit_translation("テスト", f"测试{bad_ch}"))

        # Exact leading full-width indent count preserved
        u_ind = {"kind": "narration", "speaker": None, "text": "\u3000\u3000\u3000静か。"}
        cn_ind, _, _ = runner.normalize_output_line("\u3000安静。", u_ind, [])
        self.assertTrue(cn_ind.startswith("\u3000\u3000\u3000"))

        # Glossary substitution on residual kana & overrides.jsonl pass-through without GPU
        gl = [
            {"src": "ポルカ", "dst": "波尔卡", "note": ""},
            {"src": "グリフォン", "dst": "格里芬", "note": ""},
            {"src": "ペガサス", "dst": "天马", "note": ""},
            {"src": "ホチキス", "dst": "订书机", "note": ""},
        ]
        u = {"kind": "dialogue", "speaker": "ユイ", "text": "グリフォンだよ！"}
        cn, _, flags = runner.normalize_output_line("这是グリフォン哦！", u, ["ユイ"], matched_glossary=gl)
        self.assertEqual(cn, "这是格里芬哦！")
        self.assertIn("glossary_subst", flags)
        self.assertEqual(runner.validate_unit_translation(u["text"], cn), [])

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            drive_root = root / "drive" / schema.DRIVE_ROOT_NAME
            dirs = schema.ensure_drive_tree(drive_root)
            rows = [
                {"id": "k1", "file": "f1", "kind": "ui", "text": "ポルカ"},
                {"id": "k2", "file": "f1", "kind": "ui", "text": "ホチキス"},
            ]
            inp = dirs["sakura_input"] / "kana_job.jsonl"
            inp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            schema.atomic_write_json(
                schema.manifest_path_for(inp),
                schema.build_manifest("kana_job", inp, len(rows)),
            )
            ov = dirs["sakura_input"] / "kana_job.overrides.jsonl"
            ov.write_text(
                json.dumps({"id": "k1", "translation": "波尔卡"}, ensure_ascii=False) + "\n"
                + json.dumps({"kind": "ui", "text": "ホチキス", "translation": "訂書機"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            # All resolved via overrides -> zero HTTP/GPU calls needed
            rc = runner.main(
                ["--drive-root", str(drive_root), "--api-base", "http://127.0.0.1:9/v1",
                 "--skip-gpu-check", "--manifest-wait", "0"]
            )
            self.assertEqual(rc, 0)
            marker = schema.verify_done_marker(dirs["sakura_output"], "kana_job")
            self.assertEqual(marker["qa_summary"]["manual_overrides"], 2)
            self.assertEqual(marker["qa_summary"]["unique_work_units"], 0)

    def test_placeholder_repair_and_failure_record(self):
        # Encoding {{TAG_0}} -> ⟦0⟧ and decoding spaced ⟦ 0 ⟧ back to {{TAG_0}}
        enc, tokens = runner.encode_placeholders("{{TAG_0}}と{{TAG_1}}は {{TAG_2}} 並んだ！")
        self.assertEqual(enc, "⟦0⟧と⟦1⟧は ⟦2⟧ 並んだ！")
        self.assertEqual(tokens, ["{{TAG_0}}", "{{TAG_1}}", "{{TAG_2}}"])
        dec, pflags = runner.decode_placeholders("排好了⟦ 0 ⟧的⟦1⟧共⟦2⟧！", tokens)
        self.assertEqual(dec, "排好了{{TAG_0}}的{{TAG_1}}共{{TAG_2}}！")
        self.assertIn("repair_placeholder", pflags)

        # Dropped/reordered placeholder triggers targeted retry, then writes actionable failure record
        attempts_log: list[str] = []

        class PhHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                req_obj = json.loads(self.rfile.read(length).decode("utf-8"))
                attempts_log.append(req_obj["messages"][0]["content"])
                user_msg = req_obj["messages"][-1]["content"]
                marker = "将下面的文本从日文翻译成简体中文："
                src_block = user_msg.split(marker, 1)[1][1:].strip()
                # Succeed for good_unit, always drop ⟦0⟧ for bad_unit
                reply = "恢复成功！" if "テスト" in src_block else "包裹到了！"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                ev = json.dumps({"choices": [{"delta": {"content": reply}}]}, ensure_ascii=False)
                self.wfile.write(f"data: {ev}\n\n".encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        srv = _ThreadedHTTPServer(("127.0.0.1", 0), PhHandler)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                drive_root = root / "drive" / schema.DRIVE_ROOT_NAME
                dirs = schema.ensure_drive_tree(drive_root)
                rows = [{"id": "p1", "file": "sys", "kind": "system", "text": "{{TAG_0}}が届いた！"}]
                inp = dirs["sakura_input"] / "ph_job.jsonl"
                inp.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n", encoding="utf-8")
                schema.atomic_write_json(
                    schema.manifest_path_for(inp),
                    schema.build_manifest("ph_job", inp, 1),
                )
                rc = runner.main(
                    ["--drive-root", str(drive_root), "--api-base", f"http://127.0.0.1:{port}/v1",
                     "--skip-gpu-check", "--manifest-wait", "0"]
                )
                self.assertEqual(rc, runner.EXIT_PARTIAL)
                self.assertEqual(len(attempts_log), 3)
                self.assertIn("占位符", attempts_log[1])
                fails = [
                    json.loads(ln)
                    for ln in (dirs["sakura_work"] / "ph_job.failures.jsonl").read_text("utf-8").splitlines()
                    if ln.strip()
                ]
                self.assertEqual(len(fails), 1)
                f0 = fails[0]
                self.assertEqual(f0["attempts"], 3)
                self.assertEqual(f0["expected_placeholders"], ["{{TAG_0}}"])
                self.assertEqual(f0["observed_placeholders"], [])
                self.assertEqual(f0["source_kind"], "system")
                self.assertTrue(f0["suggested_fix"])
                self.assertEqual(len(f0["unit_key"]), 40)
                self.assertFalse(schema.done_marker_path(dirs["sakura_output"], "ph_job").is_file())
                self.assertTrue((dirs["sakura_output"] / "ph_job.zh_CN.jsonl").is_file())
                self.assertTrue((dirs["sakura_output"] / "ph_job.zh_TW.jsonl").is_file())
                self.assertTrue((dirs["sakura_output"] / "ph_job.qa_report.json").is_file())
        finally:
            srv.shutdown()
            srv.server_close()

    def test_multiline_segment_split_rejoin(self):
        seen_prompts: list[str] = []
        fail_seg2_once = [True]

        class MlHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                req_obj = json.loads(self.rfile.read(length).decode("utf-8"))
                user_msg = req_obj["messages"][-1]["content"]
                src_block = user_msg.split("将下面的文本从日文翻译成简体中文：", 1)[1][1:].strip()
                seen_prompts.append(src_block)
                reply_lines = [
                    "邮递员已经到了！" if "郵便屋" in ln else "现在贴上邮票！"
                    for ln in src_block.splitlines()
                ]
                # Merge both segments once: the wrong line count must be retried
                if fail_seg2_once[0]:
                    fail_seg2_once[0] = False
                    reply_lines = ["".join(reply_lines)]
                reply = "\n".join(reply_lines)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                ev = json.dumps({"choices": [{"delta": {"content": reply}}]}, ensure_ascii=False)
                self.wfile.write(f"data: {ev}\n\n".encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        srv = _ThreadedHTTPServer(("127.0.0.1", 0), MlHandler)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            u = {
                "unit_key": runner.make_unit_key("system", None, "郵便屋が到着した！\n切手を貼る！"),
                "wid": "w000001",
                "file": "item",
                "kind": "system",
                "speaker": None,
                "text": "郵便屋が到着した！\n切手を貼る！",
            }
            cn, _, _, fail = runner._translate_single_multiline(
                f"http://127.0.0.1:{port}/v1", u, [], [], []
            )
            self.assertIsNone(fail)
            self.assertEqual(cn, "邮递员已经到了！\n现在贴上邮票！")
            # Segments are separate prompt lines of one request; a merged reply is retried
            self.assertEqual([len(p.splitlines()) for p in seen_prompts], [2, 2])
        finally:
            srv.shutdown()
            srv.server_close()

    def test_language_qa_no_cjk_and_allowlist(self):
        # Must fail no_cjk + latin_only
        self.assertEqual(runner.check_language_qa("ロードメニュー", "New Game"), ["no_cjk", "latin_only"])
        self.assertEqual(runner.check_language_qa("「……グッドラック！」", "「……Good luck！」"), ["no_cjk", "latin_only"])
        # Lost body leaving only punctuation/whitespace must fail no_cjk
        self.assertEqual(runner.check_language_qa("\u3000並べているのかな？」", "\u3000」"), ["no_cjk"])
        # Command example preserving verbatim ASCII tokens in source is allowlisted
        self.assertEqual(
            runner.check_language_qa("例：「DemoCmd SetMode 4 1」", "「DemoCmd SetMode 4 1」"),
            [],
        )

    def test_stable_unit_key_identity_and_legacy_migration(self):
        state = _MockState(fail_batch_once=False)
        srv = _ThreadedHTTPServer(("127.0.0.1", 0), _make_handler(state))
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            api_base = f"http://127.0.0.1:{port}/v1"
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                drive_root = root / "drive" / schema.DRIVE_ROOT_NAME
                dirs = schema.ensure_drive_tree(drive_root)
                rows = [
                    {"id": f"r{i:02d}", "file": "ch01", "kind": "narration", "text": f"文テスト_{i}"}
                    for i in range(10)
                ]
                # Pre-populate checkpoint with 8 units using unit_key (0..7)
                ckpt = dirs["sakura_work"] / "stable_job.checkpoint.jsonl"
                ckpt_rows = []
                for i in range(8):
                    uk = runner.make_unit_key("narration", None, rows[i]["text"])
                    ckpt_rows.append(
                        json.dumps({"unit_key": uk, "wid": f"w{i+1:06d}", "zh_cn": f"哨兵译文_{i}"}, ensure_ascii=False)
                    )
                ckpt.write_text("\n".join(ckpt_rows) + "\n", encoding="utf-8")

                # Now add an override on r00 (shifting sequential wid for r01..r09!)
                rows[0]["override"] = "手動上書き"
                inp = dirs["sakura_input"] / "stable_job.jsonl"
                inp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
                schema.atomic_write_json(
                    schema.manifest_path_for(inp),
                    schema.build_manifest("stable_job", inp, len(rows)),
                )

                rc = runner.main(
                    ["--drive-root", str(drive_root), "--api-base", api_base,
                     "--skip-gpu-check", "--manifest-wait", "0", "--force"]
                )
                self.assertEqual(rc, 0)
                # Only 2 missing units (r08, r09) should have been translated (1 batch call)!
                self.assertEqual(state.calls, 1)
                tw_rows = [
                    json.loads(ln)
                    for ln in (dirs["sakura_output"] / "stable_job.zh_TW.jsonl").read_text("utf-8").splitlines()
                    if ln.strip()
                ]
                self.assertEqual([r["id"] for r in tw_rows], [r["id"] for r in rows])
                self.assertEqual(tw_rows[0]["translation"], "手動上書き")
                for i in range(1, 8):
                    self.assertEqual(tw_rows[i]["translation"], f"哨兵譯文_{i}")
        finally:
            srv.shutdown()
            srv.server_close()

    def test_repair_only_mode_2215_cached_20_todo(self):
        state = _MockState(fail_batch_once=False)
        srv = _ThreadedHTTPServer(("127.0.0.1", 0), _make_handler(state))
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                drive_root = root / "drive" / schema.DRIVE_ROOT_NAME
                dirs = schema.ensure_drive_tree(drive_root)
                rows = [
                    {"id": f"id_{i:05d}", "file": "main", "kind": "narration", "text": f"台詞テスト_{i}"}
                    for i in range(2235)
                ]
                # Legacy wid-only checkpoint for first 2215 units
                ckpt = dirs["sakura_work"] / "repair_fixture.checkpoint.jsonl"
                ckpt.write_text(
                    "".join(
                        json.dumps({"wid": f"w{i+1:06d}", "zh_cn": f"保留译文_{i}"}, ensure_ascii=False) + "\n"
                        for i in range(2215)
                    ),
                    encoding="utf-8",
                )
                inp = dirs["sakura_input"] / "repair_fixture.jsonl"
                inp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
                schema.atomic_write_json(
                    schema.manifest_path_for(inp),
                    schema.build_manifest("repair_fixture", inp, len(rows)),
                )

                buf = io.StringIO()
                with mock.patch("sys.stdout", buf):
                    rc = runner.main(
                        ["--drive-root", str(drive_root), "--api-base", f"http://127.0.0.1:{port}/v1",
                         "--repair-only", "--job", "repair_fixture",
                         "--skip-gpu-check", "--manifest-wait", "0"]
                    )
                self.assertEqual(rc, 0)
                out_log = buf.getvalue()
                self.assertIn("cached=2215 | todo=20", out_log)
                marker = schema.verify_done_marker(dirs["sakura_output"], "repair_fixture")
                self.assertEqual(marker["qa_summary"]["total_records"], 2235)
                self.assertEqual(marker["qa_summary"]["failures"], 0)
                cn_rows = [
                    json.loads(ln)
                    for ln in (dirs["sakura_output"] / "repair_fixture.zh_CN.jsonl").read_text("utf-8").splitlines()
                    if ln.strip()
                ]
                self.assertEqual(len(cn_rows), 2235)
                for i in range(2215):
                    self.assertEqual(cn_rows[i]["translation"], f"保留译文_{i}")
        finally:
            srv.shutdown()
            srv.server_close()

    def test_done_gate_blocks_on_failures_or_no_cjk(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            drive_root = root / "drive" / schema.DRIVE_ROOT_NAME
            dirs = schema.ensure_drive_tree(drive_root)
            rows = [{"id": "d1", "file": "f1", "kind": "system", "text": "ロードメニュー"}]
            inp = dirs["sakura_input"] / "bad_lang.jsonl"
            inp.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n", encoding="utf-8")
            schema.atomic_write_json(
                schema.manifest_path_for(inp),
                schema.build_manifest("bad_lang", inp, 1),
            )
            # Legacy checkpoint containing un-translated English output
            ckpt = dirs["sakura_work"] / "bad_lang.checkpoint.jsonl"
            ckpt.write_text(json.dumps({"wid": "w000001", "zh_cn": "New Game"}) + "\n", encoding="utf-8")

            rc = runner.main(
                ["--drive-root", str(drive_root), "--assemble-only",
                 "--job", "bad_lang", "--manifest-wait", "0"]
            )
            self.assertEqual(rc, runner.EXIT_PARTIAL)
            self.assertFalse(schema.done_marker_path(dirs["sakura_output"], "bad_lang").is_file())
            self.assertTrue((dirs["sakura_output"] / "bad_lang.zh_CN.jsonl").is_file())
            self.assertTrue((dirs["sakura_output"] / "bad_lang.zh_TW.jsonl").is_file())
            qa = json.loads((dirs["sakura_output"] / "bad_lang.qa_report.json").read_text("utf-8"))
            self.assertEqual(qa["cache_rejected_units"], 1)
            self.assertEqual(qa["kept_source_rows"], 1)
            self.assertTrue(qa["over_untranslated_limit"])
            cn = json.loads((dirs["sakura_output"] / "bad_lang.zh_CN.jsonl").read_text("utf-8"))
            self.assertEqual(cn["translation"], "ロードメニュー")

    def test_checkpoint_io_retry_and_gpu_log_resilience(self):
        import builtins

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ckpt = root / "retry.checkpoint.jsonl"
            gpu_log = root / "retry.gpu.log"
            real_open = builtins.open
            calls = [0]

            def flaky_open(file, *args, **kwargs):
                if Path(file) == ckpt and calls[0] < 2:
                    calls[0] += 1
                    raise OSError("transient Drive FUSE stall")
                return real_open(file, *args, **kwargs)

            with mock.patch("builtins.open", side_effect=flaky_open), mock.patch.object(runner.time, "sleep"):
                runner._append_checkpoint_rows(
                    ckpt,
                    [{"unit_key": "k1", "wid": "w000001", "zh_cn": "测试成功"}],
                    retries=5,
                    base_delay=0.01,
                )
            self.assertEqual(calls[0], 2)
            lines = [json.loads(ln) for ln in ckpt.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertEqual(len(lines), 1)
            self.assertEqual(lines[0]["zh_cn"], "测试成功")

            def always_fail_open(file, *args, **kwargs):
                if Path(file) in (ckpt, gpu_log):
                    raise OSError("persistent FUSE error")
                return real_open(file, *args, **kwargs)

            with mock.patch("builtins.open", side_effect=always_fail_open), mock.patch.object(runner.time, "sleep"):
                with self.assertRaises(OSError):
                    runner._append_checkpoint_rows(
                        ckpt,
                        [{"unit_key": "k2", "wid": "w000002", "zh_cn": "不会写入"}],
                        retries=3,
                        base_delay=0.01,
                    )
                # GPU progress telemetry OSError must log warning and NOT crash
                runner._log_gpu_progress(1, 10, gpu_log, skip_gpu_check=True)

    def test_midrun_server_crash_returns_fatal_not_partial(self):
        class _DeadProc:
            def __init__(self):
                self.returncode = -9

            def poll(self):
                return self.returncode

            def terminate(self):
                return None

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                return None

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            drive_root = root / "drive" / schema.DRIVE_ROOT_NAME
            dirs = schema.ensure_drive_tree(drive_root)
            rows = [{"id": "c1", "file": "f1", "kind": "dialogue", "text": "「テスト」"}]
            inp = dirs["sakura_input"] / "crash_job.jsonl"
            inp.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n", encoding="utf-8")
            schema.atomic_write_json(
                schema.manifest_path_for(inp),
                schema.build_manifest("crash_job", inp, 1),
            )
            with mock.patch.object(runner, "query_gpu_stats", return_value={"name": "NVIDIA L4", "memory_total_mib": 23034.0}), \
                 mock.patch.object(runner, "ensure_llama_server", return_value=Path("/content/bin/llama-server")), \
                 mock.patch.object(runner, "stage_model", return_value=Path("/content/models/Sakura.gguf")), \
                 mock.patch.object(runner, "start_and_verify_server", return_value=(_DeadProc(), "http://127.0.0.1:9/v1")), \
                 mock.patch.object(runner, "translate_job", return_value={"job": "crash_job", "failures": 1, "missing": 1, "no_cjk": 0, "latin_only": 0}):
                with self.assertRaises(RuntimeError) as ctx:
                    runner.main(["--drive-root", str(drive_root), "--manifest-wait", "0"])
                self.assertIn("llama-server exited unexpectedly", str(ctx.exception))

            with self.assertRaises(FileNotFoundError):
                runner.main(["--drive-root", str(drive_root), "--job", "nonexistent_job", "--skip-gpu-check"])

    def test_notebook_exit_code_disconnect_matrix(self):
        def _sim_should_disconnect(rc: int, auto_disconnect: bool, disconnect_on_error: bool) -> bool:
            return (rc in (0, 2) and auto_disconnect) or (rc not in (0, 2) and disconnect_on_error)

        # AUTO_DISCONNECT=True, DISCONNECT_ON_ERROR=False (default):
        # rc=0 and rc=2 auto-disconnect to save L4 credits; rc=1 and rc=-9 retain VM
        self.assertTrue(_sim_should_disconnect(0, True, False))
        self.assertTrue(_sim_should_disconnect(2, True, False))
        self.assertFalse(_sim_should_disconnect(1, True, False))
        self.assertFalse(_sim_should_disconnect(-9, True, False))
        # AUTO_DISCONNECT=False, DISCONNECT_ON_ERROR=False -> never auto-disconnect
        self.assertFalse(_sim_should_disconnect(2, False, False))
        # DISCONNECT_ON_ERROR=True -> errors also disconnect
        self.assertTrue(_sim_should_disconnect(1, True, True))

    def test_no_duplicate_or_lost_ids(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            drive_root = root / "drive" / schema.DRIVE_ROOT_NAME
            dirs = schema.ensure_drive_tree(drive_root)
            rows = [
                {"id": "a1", "file": "f1", "kind": "narration", "text": "テストA"},
                {"id": "a2", "file": "f1", "kind": "narration", "text": "テストA"},  # duplicate text, distinct id
                {"id": "a3", "file": "f1", "kind": "ui", "text": "B", "override": "乙"},
            ]
            ckpt = dirs["sakura_work"] / "id_job.checkpoint.jsonl"
            uk = runner.make_unit_key("narration", None, "テストA")
            ckpt.write_text(json.dumps({"unit_key": uk, "wid": "w000001", "zh_cn": "测试A"}) + "\n", encoding="utf-8")
            inp = dirs["sakura_input"] / "id_job.jsonl"
            inp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            schema.atomic_write_json(
                schema.manifest_path_for(inp),
                schema.build_manifest("id_job", inp, len(rows)),
            )
            rc = runner.main(
                ["--drive-root", str(drive_root), "--api-base", "http://127.0.0.1:9/v1",
                 "--repair-only", "--skip-gpu-check", "--manifest-wait", "0"]
            )
            self.assertEqual(rc, 0)
            tw_rows = [
                json.loads(ln)
                for ln in (dirs["sakura_output"] / "id_job.zh_TW.jsonl").read_text("utf-8").splitlines()
                if ln.strip()
            ]
            out_ids = [r["id"] for r in tw_rows]
            self.assertEqual(out_ids, ["a1", "a2", "a3"])
            self.assertEqual(len(set(out_ids)), len(out_ids))

    # ---- guards against broken output ---------------------------------------

    def test_truncated_output_is_never_accepted(self):
        with _reply_server(lambda lines, n: ("啊啊啊", "length")) as (api, prompts):
            with self.assertRaises(runner.OutputTruncated):
                runner.chat_stream(api, runner.MODEL_ALIAS, [{"role": "user", "content": _MARK + "\nあ"}])
            self.assertEqual(len(prompts), 1, "truncation must not be retried as a transport error")
            unit = _unit("切手帳を閉じる")
            cn, _, _, fail = runner._translate_single_unit(api, unit, [], [], [])
        self.assertIsNone(cn)
        self.assertEqual(fail["errors"], ["truncated"])

    def test_control_codes_never_reach_model_and_survive_byte_exact(self):
        src = "\u3000「灯台っ_w!10,0,\u3000\u3000灯台っ_w!10,0,\u3000と鴎が鳴く_w!10,0,\u3000」"
        self.assertEqual("".join(p for _, p in runner.split_segments(src)), src)
        table = {"「灯台っ": "「灯塔", "灯台っ": "灯塔", "と鴎が鳴く": "海鸥在叫"}
        with _reply_server(lambda lines, n: ("\n".join(table[ln.strip()] for ln in lines), "stop")) as (api, prompts):
            cn, _, _, fail = runner._translate_single_unit(api, _unit(src), [], [], [])
        self.assertIsNone(fail)
        self.assertEqual(
            cn, "\u3000「灯塔_w!10,0,\u3000\u3000灯塔_w!10,0,\u3000海鸥在叫_w!10,0,\u3000」"
        )
        self.assertEqual(len(prompts), 1)
        self.assertNotIn("_w!", prompts[0])
        # a damaged or dropped code is rejected, e.g. for rows already in a checkpoint
        for bad in ("啊_ w!10,0,", "啊_w!10,0", "啊"):
            self.assertIn("control_code_mismatch", runner.validate_unit_translation("あ_w!10,0,", bad))
        # letters inside a control code are not English output
        self.assertEqual(runner.check_language_qa("っ_w!10,0,\u3000」", "唔_w!10,0,\u3000」"), [])

    def test_runaway_repetition_rejected_and_kept_out_of_history(self):
        self.assertIn("repetition", runner.validate_unit_translation("ええええっ、切手が", "啊" * 40 + "，邮票"))
        self.assertIn("repetition", runner.validate_unit_translation("灯台守は郵便屋の弟だ", "好棒" * 9))
        self.assertIn("too_long", runner.check_runaway("はい", "是的" + "，真的是这样没有错" * 4))
        # legitimate long moans and ordinary lines still pass
        self.assertEqual(runner.validate_unit_translation("あ" * 30, "啊" * 30), [])
        self.assertEqual(runner.validate_unit_translation("ああああっ", "啊啊啊啊啊啊啊啊啊"), [])
        # small + full-size kana are one long sound; a Chinese rendering may use one character
        self.assertEqual(
            runner.validate_unit_translation("ぷわあああああああぁぁぁぁぁぁっ！", "噗" + "哇" * 20 + "！"), []
        )
        self.assertEqual(runner.compress_long_runs("お" * 25 + "ぉ" * 9 + "っ"), "お" * 6 + "ぉ" * 6 + "っ")

        units = [_unit(f"台詞その{i}", wid=i) for i in range(2)]
        with _reply_server(lambda lines, n: ("\n".join("啊" * 200 for _ in lines), "stop")) as (api, _):
            ok, fails, hist = runner._translate_batch_recursive(api, units, [], ["之前的台词"], [])
        self.assertEqual((ok, len(fails), hist), ({}, 2, ["之前的台词"]))

    def test_interjection_rule_only_after_real_model_failure(self):
        self.assertEqual(runner.interjection_fallback("っ_w!10,0,\u3000」"), "唔_w!10,0,\u3000」")
        self.assertEqual(runner.interjection_fallback("…っ…」"), "…唔…」")
        self.assertEqual(runner.interjection_fallback("えぇぇっ？」"), "诶诶诶？」")
        self.assertIsNone(runner.interjection_fallback("うみだっ"))
        self.assertIsNone(runner.interjection_fallback("今っ"))

        echo = lambda lines, n: ("\n".join(lines), "stop")
        with _reply_server(echo) as (api, prompts):
            cn, _, flags, fail = runner._translate_single_unit(api, _unit("っ_w!10,0,\u3000」"), [], [], [])
        self.assertEqual((cn, fail), ("唔_w!10,0,\u3000」", None))
        self.assertIn("rule_fallback", flags)
        self.assertEqual(len(prompts), 3, "the model gets all three tries first")

        with mock.patch.object(runner.time, "sleep"):
            cn, _, _, fail = runner._translate_single_unit(
                "http://127.0.0.1:9/v1", _unit("っ"), [], [], []
            )
        self.assertIsNone(cn)
        self.assertTrue(fail["errors"][0].startswith("transport:"))

    def test_segment_retry_rescues_mixed_unit(self):
        src = "えっ_w!10,0,\u3000郵便屋だ"

        def reply(lines, n):
            if len(lines) > 1:
                return "合并成一行", "stop"
            return ("啊" if lines[0].startswith("え") else "是邮递员"), "stop"

        with _reply_server(reply) as (api, _):
            cn, _, flags, fail = runner._translate_single_unit(api, _unit(src), [], [], [])
        self.assertIsNone(fail)
        self.assertEqual(cn, "啊_w!10,0,\u3000是邮递员")
        self.assertIn("segment_retry", flags)

    def test_untranslated_limit_review_file_and_zero_gpu_rebuild(self):
        self.assertEqual(runner.untranslated_limit(31896, 0.001), 31)
        self.assertEqual(runner.untranslated_limit(999, 0.001), 0)
        # 国 would become 國 if kept Japanese were pushed through s2t
        hard = "中国の話"
        for extra_hard, want_rc in ((0, runner.EXIT_OK), (1, runner.EXIT_PARTIAL)):
            with tempfile.TemporaryDirectory() as td:
                dirs = schema.ensure_drive_tree(Path(td) / schema.DRIVE_ROOT_NAME)
                rows = [{"id": f"n{i}", "kind": "narration", "text": f"文{i % 5}です"} for i in range(1000)]
                rows[3]["text"] = hard
                if extra_hard:
                    rows[7]["text"] = hard + "だ"
                _write_job(dirs, "lim", rows)

                def reply(lines, n):
                    if any(hard in ln for ln in lines):
                        return "\n".join(lines), "stop"  # echoes kana -> always rejected
                    return "\n".join("译文" for _ in lines), "stop"

                buf = io.StringIO()
                with _reply_server(reply) as (api, _), mock.patch("sys.stdout", buf):
                    rc = runner.main(["--drive-root", str(dirs["root"]), "--api-base", api,
                                      "--skip-gpu-check", "--manifest-wait", "0"])
                self.assertEqual(rc, want_rc)
                self.assertNotIn("units failed", buf.getvalue())
                done = schema.done_marker_path(dirs["sakura_output"], "lim")
                self.assertEqual(done.is_file(), want_rc == runner.EXIT_OK)
                out = dirs["sakura_output"]
                tw = [json.loads(x) for x in (out / "lim.zh_TW.jsonl").read_text("utf-8").splitlines()]
                self.assertEqual(tw[3]["translation"], hard)
                review = [json.loads(x) for x in (out / "lim.review.jsonl").read_text("utf-8").splitlines()]
                self.assertEqual([r["status"] for r in review], ["kept_source"] * (1 + extra_hard))
                self.assertEqual(review[0]["first_line"], 4)
                attempts = (dirs["sakura_work"] / "lim.attempts.jsonl").read_text("utf-8")
                self.assertIn('"raw": "' + hard, attempts)

                # Fill in only the first review row, paste the file back as overrides, no server
                review[0]["translation"] = "中国的事"
                (dirs["sakura_input"] / "lim.overrides.jsonl").write_text(
                    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in review), encoding="utf-8"
                )
                rc = runner.main(["--drive-root", str(dirs["root"]), "--assemble-only",
                                  "--job", "lim", "--manifest-wait", "0"])
                # one unfilled row at most remains, which is within the limit of 1
                self.assertEqual(rc, runner.EXIT_OK)
                qa = json.loads((out / "lim.qa_report.json").read_text("utf-8"))
                self.assertEqual((qa["kept_source_rows"], qa["manual_overrides"]), (extra_hard, 1))
                self.assertTrue((dirs["sakura_work"] / "lim.attempts.jsonl").is_file())

    def test_server_death_blocks_done_even_inside_limit(self):
        with tempfile.TemporaryDirectory() as td:
            dirs = schema.ensure_drive_tree(Path(td) / schema.DRIVE_ROOT_NAME)
            rows = [{"id": f"s{i}", "kind": "narration", "text": f"文{i}です"} for i in range(2000)]
            _write_job(dirs, "dead", rows)
            # all but one unit cached; the last one hits an unreachable server
            (dirs["sakura_work"] / "dead.checkpoint.jsonl").write_text(
                "".join(
                    json.dumps({"unit_key": runner.make_unit_key("narration", None, r["text"]), "zh_cn": "译文"}) + "\n"
                    for r in rows[:-1]
                ),
                encoding="utf-8",
            )
            with mock.patch.object(runner.time, "sleep"):
                rc = runner.main(["--drive-root", str(dirs["root"]), "--api-base", "http://127.0.0.1:9/v1",
                                  "--skip-gpu-check", "--manifest-wait", "0"])
            self.assertEqual(rc, runner.EXIT_PARTIAL)
            self.assertFalse(schema.done_marker_path(dirs["sakura_output"], "dead").is_file())
            qa = json.loads((dirs["sakura_output"] / "dead.qa_report.json").read_text("utf-8"))
            self.assertEqual((qa["kept_source_rows"], qa["kept_source_limit_rows"]), (1, 2))
            self.assertEqual(qa["transport_failed_units"], 1)

    def test_bad_cached_rows_are_retranslated(self):
        with tempfile.TemporaryDirectory() as td:
            dirs = schema.ensure_drive_tree(Path(td) / schema.DRIVE_ROOT_NAME)
            rows = [
                {"id": "c1", "kind": "narration", "text": "えっ_w!10,0,\u3000郵便屋だ"},
                {"id": "c2", "kind": "narration", "text": "ええと、切手は"},
                {"id": "c3", "kind": "narration", "text": "切手帳を閉じる"},
            ]
            _write_job(dirs, "old", rows)
            cached = ["啊是邮递员", "啊" * 500, "合上集邮册"]
            (dirs["sakura_work"] / "old.checkpoint.jsonl").write_text(
                "".join(
                    json.dumps({"unit_key": runner.make_unit_key("narration", None, r["text"]), "zh_cn": c}) + "\n"
                    for r, c in zip(rows, cached)
                ),
                encoding="utf-8",
            )
            table = {"えっ": "啊", "郵便屋だ": "是邮递员", "ええと、切手は": "那个，邮票呢"}
            buf = io.StringIO()
            with _reply_server(lambda lines, n: ("\n".join(table[x] for x in lines), "stop")) as (api, prompts), \
                    mock.patch("sys.stdout", buf):
                rc = runner.main(["--drive-root", str(dirs["root"]), "--api-base", api, "--repair-only",
                                  "--skip-gpu-check", "--manifest-wait", "0"])
            self.assertEqual(rc, 0)
            self.assertIn("cached=1 | todo=2", buf.getvalue())
            self.assertNotIn("切手帳", "".join(prompts))
            cn = [json.loads(x)["translation"]
                  for x in (dirs["sakura_output"] / "old.zh_CN.jsonl").read_text("utf-8").splitlines()]
            self.assertEqual(cn, ["啊_w!10,0,\u3000是邮递员", "那个，邮票呢", "合上集邮册"])

    def test_unprotected_code_warning(self):
        found = schema.find_unprotected_codes(
            [{"text": "<color=red>赤</color>_w!10,0,{{TAG_0}}\\C[1]\\n"}]
        )
        self.assertEqual(sorted(found), ["</color>", "<color=red>", "\\n"])


_MARK = "将下面的文本从日文翻译成简体中文："


def _unit(text: str, wid: int = 1) -> dict:
    return {
        "unit_key": runner.make_unit_key("narration", None, text),
        "wid": f"w{wid:06d}", "file": "f", "kind": "narration", "speaker": None, "text": text,
    }


def _write_job(dirs, job: str, rows: list) -> None:
    inp = dirs["sakura_input"] / f"{job}.jsonl"
    inp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    schema.atomic_write_json(schema.manifest_path_for(inp), schema.build_manifest(job, inp, len(rows)))


@contextlib.contextmanager
def _reply_server(reply):
    """Mock SSE server. reply(source_lines, call_no) -> (text, finish_reason)."""
    prompts: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
            block = req["messages"][-1]["content"].split(_MARK, 1)[1][1:]
            prompts.append(block)
            text, finish = reply(block.split("\n"), len(prompts))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for ev in ({"choices": [{"delta": {"content": text}}]},
                       {"choices": [{"delta": {}, "finish_reason": finish}]}):
                self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")

    srv = _ThreadedHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/v1", prompts
    finally:
        srv.shutdown()
        srv.server_close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
