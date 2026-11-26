"""Comprehensive offline unit + integration tests for deploy/colab-hymt2.

Runs completely locally in temporary directories using a mock SSE OpenAI server.
Leaves zero artifacts in the repository.
"""
from __future__ import annotations

import ast
import http.server
import io
import json
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


class _MockState:
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = 0
        self.ui_saw_adult_or_context = False
        self.saw_plain_fallback = False
        self.saw_glossary_simplified = False


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
            self._verify_req_shape(req_obj)
            user_msg = req_obj["messages"][0]["content"]

            is_plain = user_msg.startswith("将以下文本翻译为简体中文")
            if is_plain:
                src_text = user_msg.split("\n\n", 1)[1]
                with state.lock:
                    state.saw_plain_fallback = True
            else:
                m = re.search(r"【待翻译文本】\n(.*?)\n【翻译任务】", user_msg, re.S)
                src_text = m.group(1) if m else ""

            with state.lock:
                state.calls += 1
                if "Save Game" in src_text and (
                    "成人向" in user_msg or "上文（仅供参考" in user_msg
                ):
                    state.ui_saw_adult_or_context = True
                if "翻译成 哥哥" in user_msg and "硬件" in user_msg:
                    state.saw_glossary_simplified = True

            if "Good morning" in src_text:
                reply = "早安。昨晚睡得好吗？"
            elif "going to cum" in src_text:
                reply = "啊……不要停，我要射了！"
            elif "*squelch*" in src_text:
                reply = "*噗嗤* *噗嗤* 哈啊……哈啊……"
            elif "trigger_fallback" in src_text and not is_plain:
                # Simulate explanatory prefix on rich prompt -> runner should retry plain template
                reply = "译文：坏掉了"
            elif "trigger_fallback" in src_text and is_plain:
                reply = "硬件坏掉了，哥哥。{{TAG_0}}"
            elif "watch_hair" in src_text:
                reply = "头发干净，干吧，后来怎么样？"
            else:
                reply = f"译：{src_text}"

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for ch in reply:
                ev = json.dumps({"choices": [{"delta": {"content": ch}}]}, ensure_ascii=False)
                self.wfile.write(f"data: {ev}\n\n".encode("utf-8"))
            end_ev = json.dumps(
                {"choices": [{"delta": {}}], "usage": {"completion_tokens": max(20, len(reply) * 2)}}
            )
            self.wfile.write(f"data: {end_ev}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        @staticmethod
        def _verify_req_shape(req_obj: dict):
            msgs = req_obj["messages"]
            if len(msgs) != 1 or msgs[0]["role"] != "user":
                raise AssertionError(f"Hy-MT2 must have 1 user message only, got {msgs}")

    return Handler


class _ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


class Hymt2ColabTests(unittest.TestCase):
    def test_schema_and_tw2sp_s2twp(self):
        self.assertEqual(runner.prompt_to_simplified("硬體"), "硬件")
        self.assertEqual(runner.to_traditional("硬件", "s2twp"), "硬體")
        self.assertEqual(runner.to_traditional("头发干净，干吧", "s2twp"), "頭髮乾淨，幹吧")

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "en.jsonl"
            row = json.dumps({"id": "1", "kind": "dialogue", "speaker": "優衣", "text": "Hello"})
            p.write_bytes(b"\xef\xbb\xbf" + (row + "\n").encode("utf-8"))
            recs = schema.validate_jsonl_file(p)
            self.assertEqual(len(recs), 1)

            bad = Path(td) / "bad.jsonl"
            bad.write_text(json.dumps({"id": "1", "kind": "unknown", "text": "x"}) + "\n", encoding="utf-8")
            with self.assertRaises(schema.SchemaError):
                schema.validate_jsonl_file(bad)

    def test_prompt_glossary_whole_word_and_mechanical_checks(self):
        gl = runner._prepare_glossary_simplified([
            {"src": "onii-chan", "dst": "哥哥", "note": ""},
            {"src": "hardware", "dst": "硬體", "note": ""},
            {"src": "cum", "dst": "射", "note": ""},
        ])
        # "accumulate" must NOT match "cum"
        msgs = runner.build_hymt2_messages(
            "Onii-chan, don't accumulate hardware!",
            "dialogue",
            gl,
            speaker="優衣",
            context=["Prev line 1"],
        )
        self.assertEqual(len(msgs), 1)
        content = msgs[0]["content"]
        self.assertIn("onii-chan 翻译成 哥哥", content)
        self.assertIn("hardware 翻译成 硬件", content)
        self.assertNotIn("cum 翻译成", content)
        self.assertIn("说话人是「优衣」", content)
        self.assertIn("简体中文", content)

        # UI suppresses STYLE_ADULT and context
        ui_msgs = runner.build_hymt2_messages("Save", "ui", gl, speaker=None, context=["Prev"])
        self.assertNotIn("成人向", ui_msgs[0]["content"])
        self.assertNotIn("上文", ui_msgs[0]["content"])

        # Mechanical checks
        self.assertIn("refusal", runner.mechanical_problems("I'm sorry, I cannot translate this.", "hi"))
        self.assertIn("thinking_block", runner.mechanical_problems("<think>x</think>好", "hi"))
        self.assertIn("explanatory_prefix", runner.mechanical_problems("译文：好", "hi"))
        self.assertIn("placeholder_mismatch", runner.mechanical_problems("好", "hi {{TAG_0}}"))
        self.assertIn("not_translated", runner.mechanical_problems("Hello world", "Hello world"))
        self.assertEqual(runner.mechanical_problems("你好{{TAG_0}}", "Hello {{TAG_0}}"), [])

    def test_gpu_offload_log_parser(self):
        self.assertEqual(
            runner.parse_gpu_offload_log("offloaded 49/49 layers to GPU"), (49, 49)
        )
        with self.assertRaises(RuntimeError):
            runner.parse_gpu_offload_log("offloaded 10/49 layers to GPU")
        runner_src = (HERE / "runner.py").read_text(encoding="utf-8")
        self.assertIn("LLAMA_ARG_LOG_VERBOSITY", runner_src)
        self.assertIn('"-lv", "4"', runner_src)

    def test_e2e_mock_sse_bridge_fallback_resume_and_qa(self):
        state = _MockState()
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
                (dirs["models"] / schema.MODELS["hymt2"]["filename"]).write_bytes(b"")

                local_work = root / "local"
                local_work.mkdir()
                inp_file = local_work / "en_sample.jsonl"
                gloss_file = local_work / "glossary.json"
                gloss_file.write_text(
                    json.dumps({"onii-chan": "哥哥", "hardware": "硬體"}, ensure_ascii=False),
                    encoding="utf-8",
                )

                rows = [
                    {"id": "001", "file": "ch01", "kind": "dialogue", "speaker": "優衣", "text": "Good morning!"},
                    {"id": "002", "file": "ch01", "kind": "dialogue", "speaker": "優衣", "text": "trigger_fallback hardware onii-chan {{TAG_0}}"},
                    {"id": "003", "file": "ch01", "kind": "dialogue", "speaker": "優衣", "text": "trigger_fallback hardware onii-chan {{TAG_0}}"},  # dup
                    {"id": "004", "file": "ch01", "kind": "narration", "text": "watch_hair line"},
                    {"id": "005", "file": "ch01", "kind": "narration", "text": "... 12345"},  # passthrough
                    {"id": "006", "file": "ch02", "kind": "ui", "text": "Save Game"},
                    {"id": "007", "file": "ch02", "kind": "ui", "text": "Load Game", "override": "讀取遊戲"},
                ]
                inp_file.write_text(
                    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8",
                )

                self.assertEqual(
                    drive_bridge.main(
                        ["--drive-root", str(drive_root), "--skip-drivefs-check", "push",
                         "--input", str(inp_file), "--glossary", str(gloss_file), "--name", "en_job"]
                    ),
                    0,
                )
                self.assertTrue((dirs["scripts_hymt2"] / "runner.py").is_file())

                # Seed partial checkpoint for 001 to verify resume
                ckpt = dirs["hymt2_work"] / "en_job.checkpoint.jsonl"
                ckpt.write_text(
                    json.dumps({"wid": "w000001", "zh_cn": "早安！"}, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

                # OS metadata files in input/ must be ignored safely
                (dirs["hymt2_input"] / "desktop.ini").write_text("[.ShellClassInfo]\n", encoding="utf-8")
                (dirs["hymt2_input"] / ".DS_Store").write_bytes(b"\x00")

                rc = runner.main(
                    ["--drive-root", str(drive_root), "--api-base", api_base,
                     "--manifest-wait", "0", "--min-tps", "1.0"]
                )
                self.assertEqual(rc, 0)
                self.assertFalse(state.ui_saw_adult_or_context)
                self.assertTrue(state.saw_plain_fallback)
                self.assertTrue(state.saw_glossary_simplified)

                marker = schema.verify_done_marker(dirs["hymt2_output"], "en_job")
                self.assertEqual(marker["qa_summary"]["missing"], 0)
                self.assertEqual(marker["qa_summary"]["failures"], 0)
                self.assertIn("发", marker["qa_summary"]["watch_chars"])

                pulled = local_work / "pulled"
                self.assertEqual(
                    drive_bridge.main(
                        ["--drive-root", str(drive_root), "--skip-drivefs-check", "pull",
                         "--name", "en_job", "--dest", str(pulled)]
                    ),
                    0,
                )
                tw_rows = [
                    json.loads(ln)
                    for ln in (pulled / "en_job.zh_TW.jsonl").read_text(encoding="utf-8").splitlines()
                    if ln.strip()
                ]
                self.assertEqual([r["id"] for r in tw_rows], [r["id"] for r in rows])
                self.assertEqual(tw_rows[1]["translation"], "硬體壞掉了，哥哥。{{TAG_0}}")
                self.assertEqual(tw_rows[2]["translation"], "硬體壞掉了，哥哥。{{TAG_0}}")
                self.assertEqual(tw_rows[3]["translation"], "頭髮乾淨，幹吧，後來怎麼樣？")
                self.assertEqual(tw_rows[4]["translation"], "... 12345")
                self.assertEqual(tw_rows[6]["translation"], "讀取遊戲")
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

        srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), BinHandler)
        srv.daemon_threads = True
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
        for nb_name in ("deploy_models_colab.ipynb", "hymt2_colab.ipynb"):
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
            if nb_name == "hymt2_colab.ipynb":
                self.assertNotIn("REBUILD_LLAMA_BIN", full_src)
                self.assertNotIn("--rebuild-bin", full_src)
            for c in code_cells:
                src = "".join(c["source"])
                cleaned = "\n".join(
                    ln for ln in src.splitlines() if not ln.lstrip().startswith(("!", "%"))
                )
                ast.parse(cleaned, filename=nb_name)
                if nb_name == "hymt2_colab.ipynb" and "Popen" in src:
                    self.assertIn("runner.log", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
