import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .config import Config
from .data import fingerprint
from .instruction import InstructionData, prompt_text
from .model import LanguageModel
from .train import choose_device


class Predictor:
    """Generate solely from trained weights; never look up canned answers."""
    def __init__(self, run, device="auto"):
        self.run = Path(run)
        path = self.run / "best.pt"
        if not path.exists():
            path = self.run / "latest.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("training_kind") != "instruction":
            raise ValueError("Model-only chat requires an instruction-trained checkpoint")
        if fingerprint(self.run / "tokenizer.json") != saved["tokenizer_sha256"]:
            raise ValueError("Tokenizer mismatch")
        self.step = saved["step"]
        self.cfg = Config(**saved["config"])
        torch.set_num_threads(self.cfg.cpu_threads)
        self.device = choose_device(device)
        self.model = LanguageModel(self.cfg)
        self.model.load_state_dict(saved["model"])
        del saved
        self.model.to(self.device).eval()
        self.tokenizer = Tokenizer.from_file(str(self.run / "tokenizer.json"))

    def reply(self, question, max_tokens=128):
        ids = [self.tokenizer.token_to_id("[BOS]")] + self.tokenizer.encode(prompt_text(question)).ids
        if len(ids) >= self.cfg.context:
            raise ValueError("Question exceeds the trained context; please shorten it")
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        result = self.model.generate(x, max_tokens, temperature=1.0, top_k=1,
                                     eos_id=self.tokenizer.token_to_id("[EOS]"),
                                     repetition_penalty=1.15, no_repeat_ngram=3)
        return self.tokenizer.decode(result[0, len(ids):].tolist(), skip_special_tokens=True).strip()


def evaluate_instruction(run, dataset, output):
    predictor = Predictor(run)
    data = InstructionData(dataset, predictor.cfg.context, 8, seed=31415)
    if data.meta["tokenizer_sha256"] != fingerprint(Path(run) / "tokenizer.json"):
        raise ValueError("Evaluation data and model tokenizer differ")
    rows = json.loads((Path(dataset) / "test.encoded.json").read_text(encoding="utf-8"))
    # Teacher-forced loss on every test example, never used in optimization/model selection.
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(rows), 8):
            x, y = data.collate(rows[start:start + 8], predictor.device)
            _, loss = predictor.model(x, y)
            tokens = int((y != -100).sum())
            total_loss += loss.item() * tokens
            total_tokens += tokens
    examples = [json.loads(line) for line in
                (Path(dataset) / "test.jsonl").read_text(encoding="utf-8").split("\n") if line.strip()]
    # Spread samples across the entire held-out split instead of selecting favorable answers.
    indices = [i * len(examples) // 8 for i in range(8)]
    samples = [{"question": examples[i]["question"], "reference": examples[i]["answer"],
                "prediction": predictor.reply(examples[i]["question"])} for i in indices]
    probes = ["สวัสดี", "ทำอะไรได้บ้าง", "ทำไมน้ำแข็งจึงลอยน้ำ",
              "RAM กับ VRAM ต่างกันอย่างไร", "ถ้ามีแอปเปิล 3 ลูก กินไป 1 ลูก เหลือกี่ลูก",
              "เขียน Python หาผลรวมของตัวเลขในรายการ", "อธิบายการเรียนรู้ของโมเดล AI แบบง่าย"]
    def probe(question):
        try:
            return {"question": question, "prediction": predictor.reply(question)}
        except ValueError as error:
            return {"question": question, "error": str(error)}

    report = {"checkpoint_step": predictor.step, "training_examples": data.meta["counts"]["train"],
              "test_examples": len(examples), "test_tokens": total_tokens, "test_loss_uses_all_examples": True,
              "test_loss": total_loss / total_tokens,
              "held_out_samples": samples,
              "open_probes": [probe(q) for q in probes],
              "limitation": "Loss is not factual accuracy. Human review is required; no all-topic ability claim."}
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"checkpoint_step": report["checkpoint_step"], "test_loss": report["test_loss"],
            "test_examples": len(examples), "output": str(output)}
