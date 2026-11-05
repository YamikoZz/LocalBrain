import base64
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from tokenizers import Tokenizer

from localbrain.assistant import answer
from localbrain.cli import emit_chat
from localbrain.config import Config
from localbrain.data import prepare
from localbrain.knowledge import Knowledge
from localbrain.model import LanguageModel
from localbrain.optim import CPUAdamW
from localbrain.sandbox import run_python
from localbrain.tools import Tools, calculate
from localbrain.train import generate, train
from localbrain.webtools import CheckedRedirect, SearchParser, html_text, public_url


class SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_greeting_skips_retrieval_web_and_renders_plain_text(self):
        knowledge = mock.Mock()
        with mock.patch("localbrain.assistant.research") as web:
            for greeting in ("hello", " Hello! ", "HI", "สวัสดีครับ"):
                result = answer(knowledge, greeting, use_web=True)
                self.assertEqual(result["mode"], "greeting")
                self.assertEqual(result["sources"], [])
            knowledge.search.assert_not_called()
            web.assert_not_called()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            emit_chat(result)
        self.assertIn("LocalBrain>", output.getvalue())
        self.assertIn("สวัสดี", output.getvalue())
        self.assertNotIn('"mode"', output.getvalue())

    def test_offloaded_adam_matches_torch_and_restores_cpu_state(self):
        left = torch.nn.Parameter(torch.tensor([0.2, -0.7, 1.4]))
        right = torch.nn.Parameter(left.detach().clone())
        ours = CPUAdamW([left], lr=0.003, weight_decay=0.02)
        reference = torch.optim.AdamW([right], lr=0.003, weight_decay=0.02)
        for i in range(8):
            gradient = torch.tensor([0.1 * (i + 1), -0.3, 0.6])
            left.grad = gradient.clone()
            right.grad = gradient.clone()
            ours.step()
            reference.step()
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-7)
        restored = CPUAdamW([left])
        restored.load_state_dict(ours.state_dict())
        self.assertEqual(restored.state[left]["exp_avg"].device.type, "cpu")
        self.assertEqual(restored.state[left]["step"], 8)

    def test_transformer_is_causal(self):
        cfg = Config(layers=2, width=32, heads=4, context=16,
                     vocab_size=300, dropout=0, checkpoint_blocks=False)
        model = LanguageModel(cfg).eval()
        a = torch.tensor([[1, 2, 3, 4, 5]])
        b = torch.tensor([[1, 2, 9, 8, 7]])
        with torch.no_grad():
            x, _ = model(a)
            y, _ = model(b)
        torch.testing.assert_close(x[:, :2], y[:, :2])
        self.assertFalse(torch.allclose(x[:, 2:], y[:, 2:]))

    def test_tokenizer_training_resume_and_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corpus = root / "corpus.txt"
            corpus.write_text(("โมเดลเรียนรู้จากข้อมูลภาษาไทย Python learns patterns.\n" * 120),
                              encoding="utf-8")
            cfg = Config(layers=1, width=32, heads=4, context=16,
                         vocab_size=300, batch_size=2, accumulation=1, steps=12,
                         learning_rate=0.003, dropout=0.1, device="cpu",
                         optimizer="cpu_adamw", checkpoint_blocks=True,
                         cpu_threads=1, eval_every=6, eval_batches=2, save_every=6)
            prepare(corpus, root / "dataset", cfg)
            tokenizer = Tokenizer.from_file(str(root / "dataset/tokenizer.json"))
            sentence = "ทดสอบภาษาไทย unseen 🐈"
            self.assertEqual(tokenizer.decode(tokenizer.encode(sentence).ids), sentence)
            with contextlib.redirect_stdout(io.StringIO()):
                train(root / "dataset", root / "full", cfg)
                partial = Config(**{**cfg.as_dict(), "steps": 6})
                train(root / "dataset", root / "resumed", partial)
                train(root / "dataset", root / "resumed", cfg, resume=True)
            full = torch.load(root / "full/latest.pt", weights_only=True)
            resumed = torch.load(root / "resumed/latest.pt", weights_only=True)
            self.assertEqual(resumed["step"], 12)
            for key in full["model"]:
                torch.testing.assert_close(full["model"][key], resumed["model"][key],
                                           rtol=0, atol=0)
            rows = [json.loads(line) for line in (root / "full/metrics.jsonl").read_text().splitlines()]
            self.assertLess(rows[-1]["train_loss"], rows[0]["train_loss"])
            self.assertIsInstance(generate(root / "full", "โมเดล", max_tokens=4, device="cpu"), str)
            with self.assertRaisesRegex(ValueError, "new output"):
                prepare(corpus, root / "dataset", cfg)

    def test_knowledge_thai_update_and_duplicate(self):
        with tempfile.TemporaryDirectory() as temp:
            store = Knowledge(Path(temp) / "knowledge.db")
            first = store.add("สมองโมเดลใช้การฝึกภาษาไทย GPU computes tensors", "local:test")
            self.assertTrue(first["changed"])
            self.assertFalse(store.add("สมองโมเดลใช้การฝึกภาษาไทย GPU computes tensors",
                                       "local:test")["changed"])
            self.assertTrue(store.search("ภาษาไทย"))
            self.assertTrue(store.search("GPU"))
            store.add("เนื้อหาใหม่เกี่ยวกับดวงอาทิตย์", "local:test")
            self.assertFalse(store.search("GPU"))
            self.assertTrue(store.search("ดวงอาทิตย์"))
            self.assertEqual(store.stats()["documents"], 1)

    def test_arithmetic_and_file_boundaries(self):
        self.assertEqual(calculate("(8 + 4) * 2 / 3"), 8)
        for bad in ["__import__('os')", "2**10000", "[1,2]", "True", "(1).__class__"]:
            with self.assertRaises((ValueError, SyntaxError)):
                calculate(bad)
        with tempfile.TemporaryDirectory() as temp:
            tools = Tools(temp)
            with self.assertRaises(ValueError):
                tools.write_file("../escape.txt", "no")
            tools.write_file("notes/hello.txt", "สวัสดี")
            self.assertEqual(tools.read_file("notes/hello.txt")["text"], "สวัสดี")
            with self.assertRaises(FileExistsError):
                tools.write_file("notes/hello.txt", "overwrite")

    def test_sandbox_never_falls_back_to_host(self):
        with mock.patch("localbrain.sandbox.shutil.which", return_value=None), \
             mock.patch("localbrain.sandbox.subprocess.Popen") as popen:
            with self.assertRaisesRegex(ValueError, "Sandbox disabled"):
                run_python("print(1)")
            popen.assert_not_called()

    def test_search_html_decodes_result_link(self):
        destination = "https://docs.python.org/3/tutorial/"
        encoded = "a1" + base64.urlsafe_b64encode(destination.encode()).decode().rstrip("=")
        parser = SearchParser()
        parser.feed('<li class="b_algo"><h2><a href="https://www.bing.com/ck/a?u='
                    + encoded + '">Python <strong>Tutorial</strong></a></h2></li>')
        self.assertEqual(parser.results, [{"url": destination, "title": "Python Tutorial"}])

    def test_web_parser_and_public_address_check(self):
        text, title = html_text("<title>Example</title><script>secret()</script><p>Visible</p>")
        self.assertEqual(title, "Example")
        self.assertIn("Visible", text)
        self.assertNotIn("secret", text)
        with self.assertRaises(ValueError):
            public_url("file:///etc/passwd")
        addresses = [(2, 1, 6, "", ("127.0.0.1", 80))]
        with mock.patch("localbrain.webtools.socket.getaddrinfo", return_value=addresses):
            with self.assertRaisesRegex(ValueError, "public"):
                public_url("http://localhost/test")
            with self.assertRaises(ValueError):
                CheckedRedirect().redirect_request(None, None, 302, "", {}, "http://localhost/")


if __name__ == "__main__":
    unittest.main()
