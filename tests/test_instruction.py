import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch

from localbrain.config import Config
from localbrain.instruction import InstructionData, prepare_instructions, read_pairs
from localbrain.model import LanguageModel
from localbrain.train import train
from localbrain.inference import Predictor, evaluate_instruction


class InstructionTests(unittest.TestCase):
    def test_dedup_split_and_assistant_only_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pairs = [{"question": f"What is item {i}?", "answer": f"Item {i} is a training example."}
                     for i in range(300)]
            pairs += [{"question": " What   is item 1? ", "answer": "duplicate"}]
            source = root / "pairs.json"
            source.write_text(json.dumps(pairs), encoding="utf-8")
            self.assertEqual(len(list(read_pairs(source))), 300)
            cfg = Config(layers=1, width=32, heads=4, context=64, vocab_size=300,
                         batch_size=8, checkpoint_blocks=False)
            prepare_instructions(source, root / "prepared", cfg)
            groups = {}
            for split in ("train", "val", "test"):
                rows = [json.loads(line) for line in
                        (root / "prepared" / f"{split}.jsonl").read_text().splitlines()]
                groups[split] = {r["group"] for r in rows}
            self.assertFalse(groups["train"] & groups["val"])
            self.assertFalse(groups["train"] & groups["test"])
            self.assertFalse(groups["val"] & groups["test"])
            data = InstructionData(root / "prepared", cfg.context, cfg.batch_size)
            self.assertNotIn("test", data.rows)
            x, y = data.batch("train", "cpu")
            self.assertTrue((y[:, 0] == -100).all())
            self.assertTrue((y != -100).any(dim=1).all())
            self.assertTrue((y == -100).any())
            model = LanguageModel(cfg)
            _, loss = model(x, y)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(model.tokens.weight.grad).all())
            cfg.steps = 2
            cfg.device = "cpu"
            cfg.eval_batches = cfg.eval_every = cfg.save_every = 1
            cfg.accumulation = 1
            cfg.cpu_threads = 1
            with contextlib.redirect_stdout(io.StringIO()):
                train(root / "prepared", root / "run", cfg)
            predictor = Predictor(root / "run", "cpu")
            self.assertIsInstance(predictor.reply("What is item 2?", max_tokens=3), str)
            checkpoint = torch.load(root / "run/latest.pt", weights_only=True)
            self.assertEqual(checkpoint["training_kind"], "instruction")
            self.assertTrue((root / "run/best.pt").is_file())
            report = evaluate_instruction(root / "run", root / "prepared", root / "evaluation.json")
            self.assertEqual(report["test_examples"], len(groups["test"]))
            evaluated = json.loads((root / "evaluation.json").read_text(encoding="utf-8"))
            self.assertTrue(evaluated["test_loss_uses_all_examples"])
            self.assertGreater(evaluated["test_tokens"], 0)

    def test_jsonl_preserves_unicode_line_separator_inside_content(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pairs.jsonl"
            rows = [{"question": "Question" + chr(0x2028) + "one", "answer": "answer"},
                    {"question": "second", "answer": "answer"}]
            path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                            encoding="utf-8")
            self.assertEqual(len(list(read_pairs(path))), 2)

    def test_first_pair_only_and_empty_invalid_records(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "messages.json"
            values = [[{"role": "user", "content": "first"},
                       {"role": "assistant", "content": "answer"},
                       {"role": "user", "content": "later"},
                       {"role": "assistant", "content": "requires history"}],
                      {"question": "", "answer": "empty"},
                      {"messages": [{"role": "system", "content": "not a standalone pair"}]}]
            path.write_text(json.dumps(values), encoding="utf-8")
            result = list(read_pairs(path))
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["question"], "first")


if __name__ == "__main__":
    unittest.main()
