import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from localbrain.chat import Chat, response
from localbrain.cli import main
from localbrain.tools import Tools


class ChatTests(unittest.TestCase):
    def test_capabilities_never_search_or_use_model(self):
        tools = mock.Mock()
        session = Chat(tools, use_web=True)
        with mock.patch("localbrain.chat.answer") as answer:
            for query in ("ทำอะไรได้บ้าง", "/help", "What can you do?"):
                result = session.handle(query)
                self.assertEqual(result["mode"], "help")
                self.assertIn("/calc", result["answer"])
            answer.assert_not_called()
            tools.knowledge.search.assert_not_called()

    def test_model_only_bypasses_all_natural_language_canned_routes(self):
        tools = mock.Mock()
        with mock.patch("localbrain.inference.Predictor") as predictor, \
             mock.patch("localbrain.chat.answer") as retrieval:
            predictor.return_value.reply.return_value = "learned output"
            session = Chat(tools, run="test-run", model_only=True)
            for query in ("hello", "ทำอะไรได้บ้าง", "unknown question"):
                result = session.handle(query)
                self.assertEqual(result["mode"], "learned_model")
                self.assertEqual(result["answer"], "learned output")
            retrieval.assert_not_called()
            tools.knowledge.search.assert_not_called()
            self.assertEqual(predictor.return_value.reply.call_count, 3)
            self.assertEqual(session.handle("/help")["mode"], "help")

    def test_pages_are_bounded_and_keep_last_answer(self):
        session = Chat(mock.Mock())
        original = "\n".join(f"บรรทัด {i}: " + "ก" * 100 for i in range(70))
        with mock.patch("localbrain.chat.answer", return_value=response(original)):
            result = session.handle("เอกสาร")
        count = 0
        while True:
            self.assertLessEqual(len(result["answer"]), 900)
            self.assertLessEqual(len(result["answer"].splitlines()), 12)
            count += 1
            if "พิมพ์ /more" not in result.get("note", ""):
                break
            result = session.handle("/more")
        self.assertGreater(count, 1)
        self.assertEqual(session.last["answer"], original)
        self.assertIn("ครบแล้ว", session.handle("/more")["answer"])

    def test_tools_and_file_import_are_real(self):
        with tempfile.TemporaryDirectory() as temp:
            session = Chat(Tools(Path(temp) / "work"))
            self.assertEqual(session.handle("/calc (8+4)*2")["answer"], "24")
            source = Path(temp) / "notes.txt"
            source.write_text("ความรู้เรื่อง VRAM ใช้เก็บข้อมูลบนการ์ดจอ", encoding="utf-8")
            session.handle("/ingest " + str(source))
            self.assertIn("เอกสาร 1", session.handle("/stats")["answer"])
            self.assertIn("การ์ดจอ", session.handle("/search VRAM")["answer"])
            args = json.dumps({"path": "test.txt", "content": "ทดสอบ"}, ensure_ascii=False)
            session.handle("/tool write_file " + args)
            self.assertEqual((Path(temp) / "work/files/test.txt").read_text(encoding="utf-8"), "ทดสอบ")

    def test_web_reports_failures_without_claiming_success(self):
        with tempfile.TemporaryDirectory() as temp:
            session = Chat(Tools(temp))
            values = [{"source": "https://example.com/", "error": "blocked"}]
            with mock.patch("localbrain.chat.research", return_value=values) as web:
                result = session.handle("/web topic")
            self.assertIn("ไม่สำเร็จ", result["answer"])
            web.assert_called_once_with(session.tools.knowledge, "topic")

    def test_cli_survives_invalid_command_then_calculates(self):
        with tempfile.TemporaryDirectory() as temp:
            output = io.StringIO()
            with mock.patch("sys.argv", ["localbrain", "--home", temp, "chat"]), \
                 mock.patch("builtins.input", side_effect=["/calc 1/0", "/calc 2+3", "/exit"]), \
                 contextlib.redirect_stdout(output):
                main()
            self.assertIn("ทำรายการไม่สำเร็จ", output.getvalue())
            self.assertIn("\n5\n", output.getvalue())

    def test_unknown_commands_do_not_become_web_queries(self):
        session = Chat(mock.Mock(), use_web=True)
        with mock.patch("localbrain.chat.answer") as answer:
            with self.assertRaises(ValueError):
                session.handle("/unknown")
            with self.assertRaises(ValueError):
                session.handle("/calc")
            answer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
