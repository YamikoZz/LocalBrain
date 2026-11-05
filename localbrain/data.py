import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .config import Config

SPECIAL = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def corpus_files(source):
    source = Path(source)
    files = [source] if source.is_file() else sorted(
        p for p in source.rglob("*") if p.suffix.lower() in {".txt", ".md"})
    if not files:
        raise ValueError("No UTF-8 .txt or .md corpus files found")
    return files


def split_text(path):
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError(f"Split files larger than 64 MiB first: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if len(text) < 100:
        raise ValueError(f"Corpus file needs at least 100 characters: {path}")
    cut = int(len(text) * 0.9)
    return text[:cut], text[cut:]


def prepare(source, output, cfg: Config):
    output = Path(output)
    if output.exists():
        raise ValueError(f"Dataset directory exists; choose a new output: {output}")
    files = corpus_files(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish only when every artifact has been written successfully.
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        folder = Path(scratch) / "dataset"
        folder.mkdir()
        tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=cfg.vocab_size, min_frequency=2,
            special_tokens=SPECIAL,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False)

        def training_texts():
            for file in files:
                train, _ = split_text(file)
                for start in range(0, len(train), 32768):
                    yield train[start:start + 32768]

        tok.train_from_iterator(training_texts(), trainer=trainer)
        tok.save(str(folder / "tokenizer.json"))
        counts = {}
        for idx, name in enumerate(("train", "val")):
            count = 0
            with (folder / f"{name}.bin").open("wb") as stream:
                for file in files:
                    text = split_text(file)[idx]
                    for start in range(0, len(text), 32768):
                        ids = tok.encode(text[start:start + 32768]).ids
                        ids.append(tok.token_to_id("[EOS]"))
                        np.asarray(ids, dtype="<u2").tofile(stream)
                        count += len(ids)
            if count <= cfg.context:
                raise ValueError(f"{name} has {count} tokens; needs more than {cfg.context}")
            counts[name] = count
        meta = {"vocab_size": tok.get_vocab_size(), "dtype": "<u2",
                "tokens": counts, "tokenizer_sha256": fingerprint(folder / "tokenizer.json"),
                "files": [{"path": str(p.resolve()), "sha256": fingerprint(p)} for p in files]}
        (folder / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(folder, output)
    return meta


class TokenData:
    def __init__(self, folder, context, batch_size, seed=42):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / "meta.json").read_text(encoding="utf-8"))
        self.context, self.batch_size = context, batch_size
        self.rng = np.random.default_rng(seed)
        self.arrays = {}
        if fingerprint(self.folder / "tokenizer.json") != self.meta["tokenizer_sha256"]:
            raise ValueError("Tokenizer fingerprint mismatch")
        for name in ("train", "val"):
            values = np.memmap(self.folder / f"{name}.bin", mode="r", dtype=self.meta["dtype"])
            if len(values) <= context or len(values) != self.meta["tokens"][name]:
                raise ValueError(f"Invalid or insufficient {name} token data")
            self.arrays[name] = values

    def batch(self, split, device):
        import torch
        values = self.arrays[split]
        starts = self.rng.integers(0, len(values) - self.context, size=self.batch_size)
        chunks = np.stack([values[s:s + self.context + 1] for s in starts]).astype(np.int64)
        return (torch.from_numpy(chunks[:, :-1].copy()).to(device),
                torch.from_numpy(chunks[:, 1:].copy()).to(device))
