"""Code-writing agent loop around a small self-trained model.

    1. local library: reuse a solution that already passed the same task
    2. sample N candidates from the model and run the tests
    3. repair rounds: show the failing code + error message, sample again
    4. (optional) web/documentation lookup hook when still failing
    5. store passing solutions back into the library (it grows with use)

The tests decide; the model's own confidence is never trusted.
"""
import contextlib
import hashlib
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .config import Config
from .data import fingerprint
from .instruction import prompt_text
from .model import LanguageModel
from .runner import run_tests
from .train import choose_device


class CodeModel:
    def __init__(self, run, device="auto", checkpoint=None):
        self.run = Path(run)
        path = Path(checkpoint) if checkpoint else self.run / "best.pt"
        if not path.exists():
            path = self.run / "latest.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if fingerprint(self.run / "tokenizer.json") != saved["tokenizer_sha256"]:
            raise ValueError("Tokenizer mismatch")
        self.step, self.kind = saved["step"], saved.get("training_kind")
        self.cfg = Config(**saved["config"])
        self.device = choose_device(device)
        self.model = LanguageModel(self.cfg)
        self.model.load_state_dict(saved["model"])
        del saved
        self.model.to(self.device).eval()
        self.tokenizer = Tokenizer.from_file(str(self.run / "tokenizer.json"))
        self.eos = self.tokenizer.token_to_id("[EOS]")

    def sample(self, question, n=1, temperature=0.8, top_k=40, max_tokens=200, greedy=False):
        ids = [self.tokenizer.token_to_id("[BOS]")] + self.tokenizer.encode(prompt_text(question)).ids
        room = self.cfg.context - len(ids)
        if room < 16:
            raise ValueError("Task description too long for the model context")
        x = torch.tensor([ids] * n, dtype=torch.long, device=self.device)
        out = self.model.generate(x, min(max_tokens, room), temperature=1.0 if greedy else temperature,
                                  top_k=1 if greedy else top_k, eos_id=self.eos)
        texts = []
        for row in out[:, len(ids):].tolist():
            if self.eos in row:
                row = row[:row.index(self.eos)]
            texts.append(clean(self.tokenizer.decode(row, skip_special_tokens=True)))
        return texts


def clean(text):
    """Return the code from a reply: the first fenced block if any, else the whole text."""
    text = text.strip("\n")
    if "```" in text:
        block = text.split("```", 1)[1]
        block = block.split("\n", 1)[1] if "\n" in block else ""
        return block.split("```")[0].rstrip() + "\n"
    return text.rstrip() + "\n"


def repair_prompt(task_prompt, code, message):
    return (task_prompt + "\nThis attempt is wrong:\n" + code.rstrip() + "\nError: " + message[:200]
            + "\nWrite a corrected version.")


class Library:
    """Solutions that passed their tests, keyed by the normalized task text."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS solutions (
                key TEXT, language TEXT, function TEXT, prompt TEXT, code TEXT,
                source TEXT, created REAL, PRIMARY KEY (key, code))""")

    @contextlib.contextmanager
    def db(self):
        conn = sqlite3.connect(self.path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    @staticmethod
    def key(language, prompt):
        return hashlib.sha256((language + "\n" + " ".join(prompt.lower().split())).encode()).hexdigest()

    def lookup(self, language, prompt):
        with self.db() as db:
            return [r[0] for r in db.execute(
                "SELECT code FROM solutions WHERE key = ? ORDER BY created DESC LIMIT 5",
                (self.key(language, prompt),))]

    def add(self, language, function, prompt, code, source):
        with self.db() as db:
            db.execute("INSERT OR IGNORE INTO solutions VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (self.key(language, prompt), language, function, prompt, code, source, time.time()))

    def export(self):
        with self.db() as db:
            return [dict(zip(("language", "function", "prompt", "code", "source"), r)) for r in
                    db.execute("SELECT language, function, prompt, code, source FROM solutions")]


def check_all(language, codes, function, tests, workers=6):
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda c: run_tests(language, c, function, tests), codes))


def solve(model, task, samples=8, repair_rounds=1, library=None, temperature=0.8, lookup=None):
    """Return dict(code, status, attempts, how). task needs prompt/language/function/tests."""
    language, function, tests, prompt = task["language"], task["function"], task["tests"], task["prompt"]
    attempts = 0
    if library:
        known = library.lookup(language, prompt)
        for code in known:
            attempts += 1
            if run_tests(language, code, function, tests)["status"] == "pass":
                return {"code": code, "status": "pass", "attempts": attempts, "how": "library"}
    question, best = prompt, None
    for round_ in range(repair_rounds + 1):
        codes = list(dict.fromkeys(model.sample(question, samples, temperature)))
        attempts += len(codes)
        results = check_all(language, codes, function, tests)
        for code, result in zip(codes, results):
            if result["status"] == "pass":
                if library:
                    library.add(language, function, prompt, code, "model")
                return {"code": code, "status": "pass", "attempts": attempts,
                        "how": "sample" if round_ == 0 else f"repair{round_}"}
            if best is None or result["passed"] > best[1]["passed"]:
                best = (code, result)
        question = repair_prompt(prompt, best[0], best[1]["message"])
        if lookup and round_ == repair_rounds - 1:
            hint = lookup(task)
            if hint:
                question = prompt + "\nReference:\n" + hint[:600]
    return {"code": best[0], "status": best[1]["status"], "attempts": attempts, "how": "failed",
            "message": best[1]["message"], "tests_passed": best[1]["passed"]}


def evaluate_code(run, output, samples=8, per_family=2, device="auto", languages=("python", "javascript")):
    """Benchmark: greedy pass@1 and pass@samples, split into seen vs unseen task families."""
    from .tasks import benchmark
    model = CodeModel(run, device)
    tasks = benchmark(per_family=per_family, languages=languages)
    rows, began = [], time.time()
    for task in tasks:
        greedy = model.sample(task["prompt"], 1, greedy=True)[0]
        g = run_tests(task["language"], greedy, task["function"], task["tests"])
        codes = model.sample(task["prompt"], samples, 0.8)
        results = check_all(task["language"], codes, task["function"], task["tests"])
        rows.append({"family": task["family"], "language": task["language"], "split": task["split"],
                     "greedy_pass": g["status"] == "pass", "greedy_status": g["status"],
                     "any_pass": any(r["status"] == "pass" for r in results),
                     "sample_pass_rate": sum(r["status"] == "pass" for r in results) / len(results),
                     "greedy_code": greedy, "greedy_message": g["message"]})

    def summary(selected):
        n = len(selected)
        return {"tasks": n,
                "pass@1_greedy": round(sum(r["greedy_pass"] for r in selected) / n, 3) if n else None,
                f"pass@{samples}": round(sum(r["any_pass"] for r in selected) / n, 3) if n else None,
                "sample_pass_rate": round(sum(r["sample_pass_rate"] for r in selected) / n, 3) if n else None}

    report = {"checkpoint_step": model.step, "run": str(run), "samples": samples,
              "seconds": round(time.time() - began, 1), "overall": summary(rows)}
    for split in ("seen", "unseen"):
        for language in languages:
            report[f"{split}/{language}"] = summary([r for r in rows if r["split"] == split
                                                     and r["language"] == language])
    report["rows"] = rows
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {k: v for k, v in report.items() if k != "rows"}
