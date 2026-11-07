import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from localbrain.coder import Library, clean, solve
from localbrain.config import Config
from localbrain.model import LanguageModel
from localbrain.optim import Muon, orthogonalize
from localbrain.runner import run_tests, toolchains
from localbrain.tasks import F, HELD_OUT, benchmark, literal, make_task, sft_examples
from localbrain.train import lr_scale, train

CASES = [{"args": [[1, 2, 3, 4]], "expected": 6}, {"args": [[]], "expected": 0}]


class RunnerTests(unittest.TestCase):
    def test_python_pass_fail_error(self):
        ok = run_tests("python", "def f(xs):\n    return sum(x for x in xs if x % 2 == 0)\n", "f", CASES)
        self.assertEqual((ok["status"], ok["passed"], ok["total"]), ("pass", 2, 2))
        bad = run_tests("python", "def f(xs):\n    return sum(xs)\n", "f", CASES)
        self.assertEqual(bad["status"], "fail")
        self.assertIn("expected 6", bad["message"])
        err = run_tests("python", "def f(xs):\n    return xs[9]\n", "f", CASES)
        self.assertIn("IndexError", err["message"])

    def test_python_static_rejections_never_execute(self):
        for code in ("import os\ndef f(x): return 0", "def f(x): return open('a').read()",
                     "def f(x): return ().__class__", "def f(x): return eval('1')",
                     "from subprocess import run\ndef f(x): return 0"):
            self.assertEqual(run_tests("python", code, "f", CASES)["status"], "rejected", code)

    def test_python_runtime_limits(self):
        loop = run_tests("python", "def f(xs):\n    while True:\n        pass\n", "f", CASES, timeout=2)
        self.assertEqual(loop["status"], "timeout")
        spam = run_tests("python", "def f(xs):\n    for i in range(10**5):\n        print(i)\n"
                         "    return 6 if xs else 0\n", "f", CASES)
        self.assertEqual(spam["status"], "pass")

    @unittest.skipUnless("javascript" in toolchains(), "Node.js not installed")
    def test_javascript(self):
        ok = run_tests("javascript", "function f(xs) { return xs.filter(x => x % 2 === 0)"
                       ".reduce((a, b) => a + b, 0); }", "f", CASES)
        self.assertEqual(ok["status"], "pass")
        self.assertEqual(run_tests("javascript", "function f(x){ return require('fs'); }",
                                   "f", CASES)["status"], "rejected")
        self.assertIn("SyntaxError", run_tests("javascript", "function f(x){ }}", "f", CASES)["message"])
        self.assertEqual(run_tests("javascript", "function f(x){ while(true){} }", "f", CASES,
                                   timeout=2)["status"], "timeout")


class TaskTests(unittest.TestCase):
    def test_reference_solutions_pass(self):
        rng = random.Random(3)
        langs = [l for l in ("python", "javascript") if l in toolchains()]
        for fam in F:
            for lang in langs:
                task = make_task(fam, lang, rng)
                for code in task["solutions"]:
                    result = run_tests(lang, code, task["function"], task["tests"])
                    self.assertEqual(result["status"], "pass", (fam["name"], lang, result["message"]))

    def test_held_out_families_never_in_training(self):
        families = {row["family"] for row in sft_examples(3000, 5)}
        self.assertFalse(families & HELD_OUT)
        splits = {t["split"] for t in benchmark(per_family=1)}
        self.assertEqual(splits, {"seen", "unseen"})

    def test_prompt_example_is_not_a_scored_test(self):
        for task in benchmark(per_family=2):
            shown = task["prompt"].split("Example: ", 1)[1]
            for t in task["tests"]:
                call = f"{task['function']}({', '.join(literal(a, task['language']) for a in t['args'])}) returns"
                self.assertFalse(shown.startswith(call), (task["family"], task["language"], shown))
            self.assertGreaterEqual(len(task["tests"]), 6, task["family"])

    def _agent_eval(self, fake_solve):
        class Fake:
            step = 0
            def __init__(self, *a, **k): pass
            def sample(self, prompt, n=1, temperature=0.8, greedy=False, **k):
                return ["def f(*a):\n    return None\n"] * n
        import localbrain.coder as coder
        saved = coder.CodeModel, coder.solve
        coder.CodeModel, coder.solve = Fake, fake_solve
        try:
            with tempfile.TemporaryDirectory() as d:
                return coder.evaluate_code(d, Path(d) / "r.json", samples=1, per_family=1,
                                           languages=("python",), agent=True, repair_rounds=0)
        finally:
            coder.CodeModel, coder.solve = saved

    def test_agent_mode_scores_on_hidden_tests(self):
        from localbrain.coder import PUBLIC_TESTS
        seen_counts = []

        def memorise_public(model, task, **kw):
            # Passes every public test by lookup table, knows nothing about the hidden ones.
            seen_counts.append(len(task["tests"]))
            table = {repr(list(t["args"])): t["expected"] for t in task["tests"]}
            code = f"_T = {table!r}\ndef {task['function']}(*a):\n    return _T.get(repr(list(a)), '__miss__')\n"
            self.assertEqual(run_tests("python", code, task["function"], task["tests"])["status"], "pass")
            return {"code": code, "status": "pass", "attempts": 1, "how": "sample"}

        report = self._agent_eval(memorise_public)
        self.assertEqual(set(seen_counts), {PUBLIC_TESTS})
        self.assertEqual(report["overall"]["agent_pass"], 0.0)
        self.assertFalse(report["prompt_example_in_tests"])
        self.assertIn("agent", report["modes"])

    def test_agent_mode_counts_correct_solutions(self):
        def reference(model, task, **kw):
            return {"code": task["solutions"][0], "status": "pass", "attempts": 1, "how": "sample"}
        self.assertEqual(self._agent_eval(reference)["overall"]["agent_pass"], 1.0)

    def test_clean_strips_fences(self):
        self.assertEqual(clean("```python\ndef f():\n    return 1\n```\nmore"), "def f():\n    return 1\n")


class ModelV2Tests(unittest.TestCase):
    def cfg(self, **kw):
        base = dict(layers=2, width=32, heads=4, context=32, vocab_size=300, dropout=0.0,
                    checkpoint_blocks=False, arch="v2")
        base.update(kw)
        return Config(**base)

    def test_kv_cache_matches_full_forward(self):
        torch.manual_seed(0)
        model = LanguageModel(self.cfg())
        for block in model.blocks:  # non-zero residual branches so the check is meaningful
            torch.nn.init.normal_(block.proj.weight, std=0.02)
            torch.nn.init.normal_(block.down.weight, std=0.02)
        model.eval()
        ids = torch.randint(0, 300, (2, 12))
        full, _ = model(ids)
        caches = [dict() for _ in model.blocks]
        first, _ = model(ids[:, :8], caches=caches)
        steps = [first[:, -1]]
        for t in range(8, 12):
            logits, _ = model(ids[:, t:t + 1], caches=caches, offset=t)
            steps.append(logits[:, -1])
        self.assertTrue(torch.allclose(full[:, 7:12], torch.stack(steps, 1), atol=1e-4))
        out = model.generate(ids[:, :4], 6, top_k=1, eos_id=None)
        self.assertEqual(out.shape, (2, 10))

    def test_old_config_defaults_to_gpt(self):
        cfg = Config(layers=1, width=32, heads=4, context=16, vocab_size=300)
        self.assertEqual(cfg.arch, "gpt")
        self.assertIn("positions.weight", LanguageModel(cfg).state_dict())

    def test_muon_and_schedule(self):
        g = torch.randn(64, 32)
        o = orthogonalize(g)
        s = torch.linalg.svdvals(o)
        self.assertTrue(0.5 < s.min() and s.max() < 1.5)
        p = torch.nn.Parameter(torch.randn(16, 8))
        opt = Muon([p], lr=0.05)
        (p ** 2).sum().backward()
        before = p.detach().clone()
        opt.step()
        self.assertFalse(torch.equal(before, p.detach()))
        cfg = self.cfg(schedule="wsd", steps=100, warmup_steps=10, decay_fraction=0.2, min_lr_ratio=0.1)
        self.assertAlmostEqual(lr_scale(5, cfg), 0.5)
        self.assertEqual(lr_scale(50, cfg), 1.0)
        self.assertAlmostEqual(lr_scale(100, cfg), 0.1)

    def test_pretrain_then_init_sft(self):
        from localbrain.data import SPECIAL
        from localbrain.instruction import prepare_instructions
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            text = "def add(a, b):\n    return a + b\n" * 400
            tok = Tokenizer(models.BPE(unk_token="[UNK]"))
            tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tok.decoder = decoders.ByteLevel()
            tok.train_from_iterator([text], trainers.BpeTrainer(
                vocab_size=300, special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                show_progress=False))
            pre = root / "pre"
            pre.mkdir()
            tok.save(str(pre / "tokenizer.json"))
            ids = np.asarray(tok.encode(text).ids, dtype="<u2")
            ids.tofile(pre / "train.bin")
            ids[:500].tofile(pre / "val.bin")
            from localbrain.data import fingerprint
            (pre / "meta.json").write_text(json.dumps({
                "kind": "language", "vocab_size": tok.get_vocab_size(), "dtype": "<u2",
                "tokens": {"train": len(ids), "val": 500},
                "tokenizer_sha256": fingerprint(pre / "tokenizer.json")}))
            cfg = self.cfg(vocab_size=tok.get_vocab_size(), context=256, steps=3, batch_size=2, accumulation=1,
                           eval_every=1, eval_batches=1, save_every=1, device="cpu", cpu_threads=1,
                           optimizer="muon", schedule="wsd", warmup_steps=1)
            with contextlib.redirect_stdout(io.StringIO()):
                train(pre, root / "run-pre", cfg)
            pairs = root / "pairs.jsonl"
            pairs.write_text("\n".join(json.dumps(r) for r in sft_examples(200, 1)), encoding="utf-8")
            prepare_instructions(pairs, root / "sft", cfg, tokenizer=pre / "tokenizer.json")
            with contextlib.redirect_stdout(io.StringIO()):
                train(root / "sft", root / "run-sft", cfg, init=root / "run-pre/latest.pt")
            saved = torch.load(root / "run-sft/latest.pt", weights_only=True)
            self.assertEqual(saved["training_kind"], "instruction")
            from localbrain.coder import CodeModel
            model = CodeModel(root / "run-sft", "cpu")
            self.assertEqual(len(model.sample("Write a Python function `f(x)` that returns x.", 3,
                                              max_tokens=8)), 3)


class LibraryTests(unittest.TestCase):
    def test_library_short_circuits_model(self):
        class Never:
            def sample(self, *a, **k):
                raise AssertionError("model should not be called")
        with tempfile.TemporaryDirectory() as temp:
            lib = Library(Path(temp) / "lib.sqlite3")
            task = {"prompt": "sum evens", "language": "python", "function": "f", "tests": CASES}
            lib.add("python", "f", "sum evens", "def f(xs):\n    return sum(x for x in xs if x % 2 == 0)\n", "test")
            result = solve(Never(), task, library=lib)
            self.assertEqual((result["status"], result["how"]), ("pass", "library"))


if __name__ == "__main__":
    unittest.main()


class HelperTests(unittest.TestCase):
    def test_offline_helpers(self):
        from localbrain.helpers import SearchCache, docs_lookup, explain_error
        self.assertIn("split", docs_lookup("str.split")["doc"])
        self.assertIn("Array/map", docs_lookup("Array.prototype.map", "javascript")["url"])
        with self.assertRaises(ValueError):
            docs_lookup("os.remove")
        self.assertIn("got 3", explain_error("f([1]) returned 3, expected 2"))
        self.assertIn("base case", explain_error("RecursionError: maximum recursion depth exceeded"))
        with tempfile.TemporaryDirectory() as temp:
            cache = SearchCache(Path(temp) / "c.sqlite3")
            cache.put("k", {"source": "x", "results": [1]})
            self.assertEqual(cache.get("k")["results"], [1])

    def test_curriculum_starts_with_short_examples(self):
        from localbrain.instruction import InstructionData
        data = InstructionData.__new__(InstructionData)
        data.rng, data.batch_size, data.curriculum, data.progress = np.random.default_rng(0), 4, True, 0.0
        data.rows = {"train": [{"ids": list(range(n)), "answer_start": 1} for n in range(3, 103)]}
        x, _ = data.batch("train", "cpu")
        self.assertLessEqual(x.size(1), 33)  # only the shortest 30% at the start
        data.progress = 1.0
        del data._by_length
        lengths = {data.batch("train", "cpu")[0].size(1) for _ in range(50)}
        self.assertGreater(max(lengths), 50)
