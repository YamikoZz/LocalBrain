"""Build a pretraining token stream from code + English parquet shards.

Code: codeparrot/github-code-clean rows, kept only for permissive licenses.
English: FineWeb-Edu rows with a high educational score.
Output matches TokenData: train.bin / val.bin (uint16), tokenizer.json, meta.json.
"""
import hashlib
import json
import os
import random
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .data import SPECIAL, fingerprint

PERMISSIVE = {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc", "unlicense", "cc0-1.0"}
# Languages we train and evaluate first get every file; others are subsampled.
FOCUS = {"Python", "JavaScript", "TypeScript"}
SKIP = {"HTML", "CSS", "TeX", "Makefile", "CMake", "Batchfile", "Dockerfile", "Visual Basic",
        "Assembly", "FORTRAN"}


def _clean_code(row, skip=SKIP):
    code = row["code"]
    if row["license"] not in PERMISSIVE or row["language"] in skip:
        return None
    if not 64 <= len(code) <= 32768:
        return None
    lines = code.split("\n")
    if max(map(len, lines)) > 400 or len(code) / max(1, len(lines)) > 120:
        return None  # minified or generated
    ascii_share = sum(c < 128 for c in code.encode("utf-8", "replace")) / max(1, len(code))
    if ascii_share < 0.97:
        return None
    return code


def iter_documents(code_files, english_files, english_share, other_share, seed):
    import pyarrow.parquet as pq
    rng = random.Random(seed)
    seen = set()

    def unique(text):
        key = hashlib.sha1(" ".join(text.split()).encode("utf-8", "replace")).digest()
        if key in seen:
            return False
        seen.add(key)
        return True

    def code_rows():
        for path in code_files:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=2048, columns=["code", "language", "license"]):
                for row in batch.to_pylist():
                    if row["language"] not in FOCUS and rng.random() > other_share:
                        continue
                    code = _clean_code(row)
                    if code and unique(code):
                        yield "code:" + row["language"], code

    def english_rows():
        for path in english_files:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=1024, columns=["text", "int_score"]):
                for row in batch.to_pylist():
                    if row["int_score"] >= 4 and 200 <= len(row["text"]) <= 20000 and unique(row["text"]):
                        yield "english", row["text"]

    code_iter, english_iter = code_rows(), english_rows()
    code_chars = english_chars = 0
    while code_iter:
        want_english = english_iter and (not code_iter or
                                         english_chars < english_share * (code_chars + english_chars + 1))
        source = english_iter if want_english else code_iter
        item = next(source, None)
        if item is None:
            if source is code_iter:
                return  # English only supplements code; stop when code runs out
            english_iter = None
            continue
        if item[0] == "english":
            english_chars += len(item[1])
        else:
            code_chars += len(item[1])
        yield item


def is_val(text):
    return hashlib.sha1(text[:2000].encode("utf-8", "replace")).digest()[0] < 2  # ~0.8%


def build_corpus(code_files, english_files, output, vocab_size=16384, max_tokens=600_000_000,
                 english_share=0.15, other_share=0.35, tokenizer_chars=150_000_000, seed=7):
    output = Path(output)
    if output.exists():
        raise ValueError("Output exists; choose a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        folder = Path(scratch) / "dataset"
        folder.mkdir()
        # 1) Tokenizer from the first documents of the (deterministic) mixed stream.
        tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL,
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                      show_progress=False)

        def sample():
            total = 0
            for _, text in iter_documents(code_files, english_files, english_share, other_share, seed):
                if is_val(text):
                    continue
                total += len(text)
                yield text
                if total >= tokenizer_chars:
                    return

        tok.train_from_iterator(sample(), trainer=trainer)
        tok.save(str(folder / "tokenizer.json"))
        eos = tok.token_to_id("[EOS]")
        # 2) Encode the full stream.
        counts, mix, docs = {"train": 0, "val": 0}, Counter(), 0
        streams = {s: (folder / f"{s}.bin").open("wb") for s in counts}
        batch = []

        def flush():
            for (split, kind), enc in zip([(s, k) for s, k, _ in batch],
                                          tok.encode_batch([t for _, _, t in batch])):
                ids = np.asarray(enc.ids + [eos], dtype="<u2")
                ids.tofile(streams[split])
                counts[split] += len(ids)
                mix[kind] += len(ids)
            batch.clear()

        try:
            for kind, text in iter_documents(code_files, english_files, english_share, other_share, seed):
                batch.append(("val" if is_val(text) else "train", kind, text))
                docs += 1
                if len(batch) >= 4096:
                    flush()
                    print(json.dumps({"docs": docs, "tokens": counts}), flush=True)
                    if counts["train"] >= max_tokens:
                        break
            flush()
        finally:
            for s in streams.values():
                s.close()
        meta = {"kind": "language", "vocab_size": tok.get_vocab_size(), "dtype": "<u2",
                "tokens": counts, "documents": docs, "mixture_tokens": dict(mix.most_common()),
                "tokenizer_sha256": fingerprint(folder / "tokenizer.json"),
                "licenses_kept": sorted(PERMISSIVE), "english_filter": "FineWeb-Edu int_score >= 4",
                "sources": [{"path": str(Path(p).resolve()), "bytes": Path(p).stat().st_size}
                            for p in list(code_files) + list(english_files)],
                "split": "SHA1 of document prefix, ~0.8% validation"}
        (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        os.replace(folder, output)
    return meta


def build_anneal(output, tokenizer, code_files, pyedu_files, english_files, max_tokens=80_000_000,
                 shares=None, seed=17, personal_file=None, personal_repeat=4):
    """High-quality mix for the final (learning-rate decay) phase, using an existing tokenizer.

    Default shares by characters: python-edu 35%, English 30%, GitHub code 20%,
    instruction-style text (coding skills + human chat) 15%.
    """
    import pyarrow.parquet as pq
    from .chatdata import code_skills
    from .instruction import prompt_text
    shares = shares or ({"pyedu": 0.25, "english": 0.25, "code": 0.25, "chat": 0.13, "personal": 0.12}
                        if personal_file else {"pyedu": 0.35, "english": 0.30, "code": 0.20, "chat": 0.15})
    output = Path(output)
    if output.exists():
        raise ValueError("Output exists; choose a new directory")
    tok = Tokenizer.from_file(str(tokenizer))
    eos = tok.token_to_id("[EOS]")
    rng = random.Random(seed)

    def pyedu():
        for path in pyedu_files:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=["text", "int_score"]):
                for row in batch.to_pylist():
                    if row["int_score"] >= 4 and 64 <= len(row["text"]) <= 32768:
                        yield row["text"]

    def english():
        for path in english_files:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=["text", "int_score"]):
                for row in batch.to_pylist():
                    if row["int_score"] >= 4 and 200 <= len(row["text"]) <= 20000:
                        yield row["text"]

    def code():
        # Every language, including web (HTML/CSS) and Lua which pretraining under-represented.
        keep_extra = {"Lua": 1.0, "HTML": 0.5, "CSS": 1.0, "PHP": 1.0, "C": 0.6, "C#": 0.6, "C++": 0.6,
                      "Python": 1.0, "JavaScript": 1.0, "TypeScript": 1.0}
        for path in code_files:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=["code", "language", "license"]):
                for row in batch.to_pylist():
                    if rng.random() > keep_extra.get(row["language"], 0.3):
                        continue
                    text = _clean_code(row, skip={"Batchfile", "Makefile", "CMake", "Dockerfile", "TeX"})
                    if text:
                        yield text

    def personal():
        from .personal import documents
        for _ in range(personal_repeat):
            docs = list(documents(personal_file))
            rng.shuffle(docs)
            yield from docs

    def chat():
        # Instruction-shaped documents teach the "User:/Assistant:" turn format before SFT.
        for row in code_skills(10**9, seed + 1):
            yield prompt_text(row["question"]) + " " + row["answer"]

    sources = {"pyedu": pyedu(), "english": english(), "code": code(), "chat": chat()}
    if personal_file:
        sources["personal"] = personal()
    used = {name: 0 for name in sources}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        folder = Path(scratch) / "dataset"
        folder.mkdir()
        (folder / "tokenizer.json").write_bytes(Path(tokenizer).read_bytes())
        counts, mix = {"train": 0, "val": 0}, Counter()
        streams = {s: (folder / f"{s}.bin").open("wb") for s in counts}
        batch = []

        def flush():
            for (split, kind), enc in zip([(s, k) for s, k, _ in batch],
                                          tok.encode_batch([t for _, _, t in batch])):
                ids = np.asarray(enc.ids + [eos], dtype="<u2")
                ids.tofile(streams[split])
                counts[split] += len(ids)
                mix[kind] += len(ids)
            batch.clear()

        try:
            while counts["train"] < max_tokens and sources:
                total = sum(used.values()) + 1
                # Pick the source furthest below its target share.
                name = min(sources, key=lambda n: used[n] / total - shares[n])
                text = next(sources[name], None)
                if text is None:
                    del sources[name]
                    continue
                used[name] += len(text)
                batch.append(("val" if is_val(text) else "train", name, text))
                if len(batch) >= 2048:
                    flush()
                    print(json.dumps({"tokens": counts, "mix": dict(mix)}), flush=True)
            flush()
        finally:
            for s in streams.values():
                s.close()
        meta = {"kind": "language", "vocab_size": tok.get_vocab_size(), "dtype": "<u2", "tokens": counts,
                "mixture_tokens": dict(mix.most_common()), "target_shares": shares,
                "tokenizer_sha256": fingerprint(folder / "tokenizer.json"),
                "purpose": "annealing (final learning-rate decay phase)",
                "sources": [str(Path(p).resolve()) for p in list(code_files) + list(pyedu_files) + list(english_files)]}
        (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        os.replace(folder, output)
    return meta
