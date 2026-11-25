"""Comprehensive offline unit + integration tests for deploy/colab-sakura.

Runs completely locally in temporary directories using a mock SSE OpenAI server.
Leaves zero artifacts in the repository.
"""
from __future__ import annotations

import ast
import http.server
import json
import os
import re
import socketserver
import sys
import tempfile
import threading
import unittest
from pathlib import Path

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
            after = user_msg[idx + len(marker):]
            src_block = after.split("\n", 1)[1] if "\n" in after else ""
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

        # Asymmetric split quotes must match source
        self.assertEqual(runner.validate_unit_translation("「今日は", "「今天也是"), [])
        self.assertIn("quote_asymmetry_mismatch", runner.validate_unit_translation("「今日は", "「今天也是」"))

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

    def test_gpu_offload_log_parser(self):
        ok_log = "llm_load_tensors: offloaded 49/49 layers to GPU\nmodel loaded"
        self.assertEqual(runner.parse_gpu_offload_log(ok_log), (49, 49))
        with self.assertRaises(RuntimeError):
            runner.parse_gpu_offload_log("llm_load_tensors: offloaded 20/49 layers to GPU")
        with self.assertRaises(RuntimeError):
            runner.parse_gpu_offload_log("no offload line")

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

    def test_notebooks_valid_json_and_python_syntax(self):
        for nb_name in ("deploy_models_colab.ipynb", "sakura_colab.ipynb"):
            nb_path = HERE / nb_name
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
            self.assertEqual(nb["nbformat"], 4)
            code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
            self.assertGreaterEqual(len(code_cells), 2)
            for c in code_cells:
                src = "".join(c["source"])
                cleaned = "\n".join(
                    ln for ln in src.splitlines() if not ln.lstrip().startswith(("!", "%"))
                )
                ast.parse(cleaned, filename=nb_name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
