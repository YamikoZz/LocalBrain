"""Import public, AI-generated code SFT datasets into LocalBrain's {"question", "answer"} JSONL.

Sources (licenses checked 2026-11-08 via the Hugging Face API):
  selfoss    bigcode/self-oss-instruct-sc2-exec-filter-50k  ODC-By   StarCoder2-generated, execution-filtered, Python
  opc-edu    OpenCoder-LLM/opc-sft-stage2 educational_instruct  MIT  Python, ships test cases
  opc-mceval OpenCoder-LLM/opc-sft-stage2 mceval_instruct       MIT  many languages; we keep Python/Lua/JS/TS/C/C++
  oci        nvidia/OpenCodeInstruct (2 of 50 shards)          CC-BY-4.0  only rows whose recorded tests all pass

Text processing only: no code from these datasets is executed here (CLAUDE.md rule 2); test results come from
the datasets' own metadata. Rows are dropped when their QUESTION shares a word 13-gram with a HumanEval/MBPP task
text (punctuation ignored; solutions are not compared, because boilerplate like "for i in range(1, n + 1)" is
shared by all code), mention a held-out task function name, contain a refusal, have no code block, or are too long.

  python -I scripts/import_code_sft.py --raw workspace/raw-hf --output workspace/raw-sft/hf-code.jsonl
"""
import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KEEP_LANGS = {"python": "python", "lua": "lua", "javascript": "javascript", "js": "javascript", "jsx": "javascript",
              "typescript": "typescript", "ts": "typescript", "tsx": "typescript",
              "c": "c", "cpp": "cpp", "c++": "cpp"}
REFUSAL = re.compile(r"\b(I'?m sorry|I cannot|I can'?t (help|assist|provide)|as an AI( language model)?)\b", re.I)
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
N = 13
MAX_CHARS = 6000


def words(text):
    return [w.lower() for w in WORD.findall(unicodedata.normalize("NFKC", text))]


def ngrams(text, n=N):
    w = words(text)
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def eval_ngrams(raw):
    """Word 13-grams of every HumanEval/MBPP task text (HumanEval prompt incl. docstring; all MBPP splits),
    plus whole texts shorter than 13 words (many MBPP tasks), matched as a word substring of the question."""
    import pyarrow.parquet as pq
    texts = [r["prompt"] for r in pq.read_table(raw / "eval" / "humaneval.parquet").to_pylist()]
    for split in ("train", "test", "validation", "prompt"):
        texts += [r["text"] for r in pq.read_table(raw / "eval" / f"mbpp-{split}.parquet").to_pylist()]
    grams, short = set(), set()
    for text in texts:
        w = words(text)
        if len(w) >= N:
            grams |= ngrams(text)
        elif len(w) >= 6:
            short.add(" " + " ".join(w) + " ")
    return grams, short


def overlaps_eval(question, contaminated):
    grams, short = contaminated
    if ngrams(question) & grams:
        return True
    padded = " " + " ".join(words(question)) + " "
    return any(s in padded for s in short)


def held_out_names():
    from localbrain.tasks import FAMILIES, HELD_OUT
    return {name for fam in HELD_OUT for name in FAMILIES[fam]["names"]}


def fence_language(answer):
    m = re.search(r"```([A-Za-z0-9_+#\-]*)", answer)
    return m.group(1).lower() if m else None


def rows(raw):
    import pyarrow.parquet as pq
    for r in pq.read_table(raw / "selfoss.parquet", columns=["instruction", "response"]).to_pylist():
        yield "selfoss", r["instruction"], r["response"]
    for r in pq.read_table(raw / "opc-edu.parquet", columns=["instruction", "output"]).to_pylist():
        yield "opc-edu", r["instruction"], r["output"]
    for r in pq.read_table(raw / "opc-mceval.parquet", columns=["instruction", "output"]).to_pylist():
        yield "opc-mceval", r["instruction"], r["output"]
    for path in sorted(raw.glob("oci-*.parquet")):
        t = pq.read_table(path, columns=["input", "output", "average_test_score"]).to_pylist()
        for r in t:
            if r["average_test_score"] == 1.0:
                yield "oci", r["input"], r["output"]
            else:
                yield "oci", None, "failed recorded tests"


def convert(raw, output):
    contaminated, names = eval_ngrams(raw), held_out_names()
    name_re = re.compile(r"\b(" + "|".join(sorted(map(re.escape, names))) + r")\s*\(")
    kept, seen = [], set()
    dropped = collections.Counter()
    for source, question, answer in rows(raw):
        if question is None:
            dropped[(source, "tests_failed")] += 1
            continue
        question, answer = question.strip(), answer.strip()
        lang = KEEP_LANGS.get(fence_language(answer) or "")
        if not question or not answer or "```" not in answer:
            reason = "no_code_block"
        elif lang is None:
            reason = "language_not_kept"
        elif len(question) + len(answer) > MAX_CHARS:
            reason = "too_long"
        elif REFUSAL.search(answer):
            reason = "refusal"
        elif name_re.search(question) or name_re.search(answer):
            reason = "held_out_name"
        elif overlaps_eval(question, contaminated):
            reason = "eval_overlap"
        else:
            key = "".join(unicodedata.normalize("NFKC", question).casefold().split())
            reason = "duplicate" if key in seen else None
            seen.add(key)
        if reason:
            dropped[(source, reason)] += 1
            continue
        kept.append({"question": question, "answer": answer, "source": source, "language": lang})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    by = collections.Counter((r["source"], r["language"]) for r in kept)
    return {"kept": len(kept), "by_source_language": {f"{s}/{l}": n for (s, l), n in sorted(by.items())},
            "by_language": dict(collections.Counter(r["language"] for r in kept).most_common()),
            "dropped": {f"{s}/{r}": n for (s, r), n in sorted(dropped.items())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw", default="workspace/raw-hf")
    parser.add_argument("--output", default="workspace/raw-sft/hf-code.jsonl")
    args = parser.parse_args()
    report = convert(Path(args.raw), Path(args.output))
    Path(args.output).with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
