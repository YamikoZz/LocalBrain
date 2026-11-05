"""Assistant-only supervised training data, split by normalized question."""
import hashlib
import json
import os
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .data import SPECIAL, fingerprint


def prompt_text(question):
    return "User: " + question.strip() + "\nAssistant:"


def read_pairs(source):
    path = Path(source)
    if path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("Instruction file exceeds 128 MiB")
    text = path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.split("\n") if line.strip()] if path.suffix == ".jsonl" else json.loads(text)
    if not isinstance(rows, list):
        raise ValueError("Expected JSON list or JSONL examples")
    seen = set()
    for row in rows:
        if isinstance(row, dict) and "question" in row and "answer" in row:
            question, answer = row["question"], row["answer"]
        else:
            messages = row.get("messages", []) if isinstance(row, dict) else row
            if not isinstance(messages, list) or len(messages) < 2:
                continue
            # Use only a standalone first user/assistant turn, not context-dependent later turns.
            if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
                continue
            question, answer = messages[0].get("content"), messages[1].get("content")
        if not isinstance(question, str) or not isinstance(answer, str):
            continue
        question, answer = question.strip(), answer.strip()
        if not question or not answer or len(question) + len(answer) > 12000:
            continue
        key = "".join(unicodedata.normalize("NFKC", question).casefold().split())
        if key in seen:
            continue
        seen.add(key)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 20
        split = "test" if bucket == 0 else "val" if bucket == 1 else "train"
        yield {"question": question, "answer": answer, "split": split, "group": digest}


def prepare_instructions(source, output, cfg, tokenizer=None):
    """tokenizer: reuse an existing tokenizer.json (required to fine-tune a pretrained run)."""
    output = Path(output)
    if output.exists():
        raise ValueError("Instruction output exists; choose a new directory")
    pairs = list(read_pairs(source))
    if not pairs:
        raise ValueError("No supported question/answer pairs")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        folder = Path(scratch) / "dataset"
        folder.mkdir()
        if tokenizer:
            tok = Tokenizer.from_file(str(tokenizer))
            (folder / "tokenizer.json").write_bytes(Path(tokenizer).read_bytes())
        else:
            tok = Tokenizer(models.BPE(unk_token="[UNK]"))
            tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tok.decoder = decoders.ByteLevel()
            trainer = trainers.BpeTrainer(
                vocab_size=cfg.vocab_size, min_frequency=2, special_tokens=SPECIAL,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
            tok.train_from_iterator(
                (prompt_text(p["question"]) + p["answer"] for p in pairs if p["split"] == "train"),
                trainer=trainer)
            tok.save(str(folder / "tokenizer.json"))
        counts, dropped = {}, {}
        for split in ("train", "val", "test"):
            encoded = []
            with (folder / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
                for pair in pairs:
                    if pair["split"] != split:
                        continue
                    prompt = [tok.token_to_id("[BOS]")] + tok.encode(prompt_text(pair["question"])).ids
                    answer = tok.encode(pair["answer"]).ids + [tok.token_to_id("[EOS]")]
                    ids = prompt + answer
                    if len(ids) > cfg.context + 1:
                        dropped[split] = dropped.get(split, 0) + 1
                        continue
                    encoded.append({"ids": ids, "answer_start": len(prompt)})
                    stream.write(json.dumps(pair, ensure_ascii=False) + "\n")
            if len(encoded) < 2:
                raise ValueError(f"Need at least two fitting examples in {split}")
            (folder / f"{split}.encoded.json").write_text(
                json.dumps(encoded, separators=(",", ":")), encoding="utf-8")
            counts[split] = len(encoded)
        meta = {"kind": "instruction", "vocab_size": tok.get_vocab_size(),
                "counts": counts, "too_long": dropped, "context": cfg.context,
                "tokenizer_sha256": fingerprint(folder / "tokenizer.json"),
                "source_sha256": fingerprint(source),
                "source": str(Path(source).resolve()),
                "split": "SHA256 of normalized question: 90/5/5; tokenizer trained on train only",
                "loss": "assistant tokens and EOS only; prompt and padding ignored"}
        (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        os.replace(folder, output)
    return meta


class InstructionData:
    def __init__(self, folder, context, batch_size, seed=42):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / "meta.json").read_text(encoding="utf-8"))
        if context != self.meta["context"]:
            raise ValueError("Instruction context differs; prepare again in a new directory")
        if fingerprint(self.folder / "tokenizer.json") != self.meta["tokenizer_sha256"]:
            raise ValueError("Tokenizer fingerprint mismatch")
        self.context, self.batch_size = context, batch_size
        self.rng = np.random.default_rng(seed)
        self.curriculum, self.progress = False, 1.0
        self.rows = {}
        # Test data is not loaded during optimization or validation.
        for name in ("train", "val"):
            self.rows[name] = json.loads(
                (self.folder / f"{name}.encoded.json").read_text(encoding="utf-8"))

    def batch(self, split, device):
        examples = self.rows[split]
        pool = len(examples)
        if self.curriculum and split == "train":
            if not hasattr(self, "_by_length"):
                self._by_length = sorted(examples, key=lambda r: len(r["ids"]))
            examples = self._by_length
            # Easy-to-hard: first 30% shortest examples, growing to everything by 60% of training.
            pool = max(self.batch_size, int(len(examples) * min(1.0, 0.3 + self.progress * 7 / 6)))
        indices = self.rng.integers(0, pool, size=self.batch_size)
        rows = [examples[i] for i in indices]
        return self.collate(rows, device)

    @staticmethod
    def collate(rows, device):
        length = max(len(row["ids"]) - 1 for row in rows)
        x = torch.zeros((len(rows), length), dtype=torch.long)
        y = torch.full_like(x, -100)
        for i, row in enumerate(rows):
            ids = row["ids"]
            size = len(ids) - 1
            x[i, :size] = torch.tensor(ids[:-1])
            labels = torch.tensor(ids[1:])
            labels[:row["answer_start"] - 1] = -100
            y[i, :size] = labels
        return x.to(device), y.to(device)
