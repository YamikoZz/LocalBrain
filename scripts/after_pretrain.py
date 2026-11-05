"""Wait for pretraining to finish, then run English SFT and the evaluations.

Runs unattended (no Claude needed). Progress goes to workspace/pipeline.log.
Usage: python scripts/after_pretrain.py
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "workspace/run-code-pretrain"
SFT = ROOT / "workspace/run-english-sft"
LOG = ROOT / "workspace/pipeline.log"
TARGET = json.loads((ROOT / "configs/code-pretrain.json").read_text())["steps"]
PROBES = ["Hi", "What can you do?", "Why is the sky blue?", "Give me three tips for learning to code.",
          "Write a Python function `is_even(n)` that returns True if n is even, otherwise False.",
          "Explain what this Python code does:\n\n```python\ndef f(xs):\n    return [x * 2 for x in xs]\n```",
          "What is the difference between a list and a tuple in Python?"]


def log(message):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def last_step():
    path = PRE / "metrics.jsonl"
    if not path.exists():
        return 0
    lines = path.read_text().splitlines()
    return json.loads(lines[-1])["step"] if lines else 0


def run(*args):
    log("run: " + " ".join(args))
    result = subprocess.run([sys.executable, "-m", "localbrain", *args], cwd=ROOT,
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    with LOG.open("a", encoding="utf-8") as f:
        f.write(result.stdout[-4000:] + result.stderr[-4000:] + "\n")
    if result.returncode:
        log(f"FAILED ({result.returncode}); stopping pipeline")
        sys.exit(1)
    return result.stdout


def main():
    log(f"waiting for pretraining to reach step {TARGET}")
    while last_step() < TARGET:
        time.sleep(120)
    time.sleep(30)  # let the final checkpoint write finish
    log("pretraining done; starting SFT")
    if not (SFT / "latest.pt").exists():
        run("train", "--data", "workspace/dataset-english-sft", "--run", str(SFT.relative_to(ROOT)),
            "--config", "configs/code-sft.json", "--init", str((PRE / "latest.pt").relative_to(ROOT)))
    run("code-eval", "--run", str(SFT.relative_to(ROOT)), "--samples", "8")
    from_root = str(ROOT)
    sys.path.insert(0, from_root)
    from localbrain.playground import load
    import torch
    from localbrain.instruction import prompt_text
    model, tok, step, _ = load(SFT)
    answers = []
    for q in PROBES:
        ids = [tok.token_to_id("[BOS]")] + tok.encode(prompt_text(q)).ids
        out = model.generate(torch.tensor([ids], device=next(model.parameters()).device), 200,
                             temperature=0.7, top_k=40, eos_id=tok.token_to_id("[EOS]"), repetition_penalty=1.1)
        answers.append({"question": q, "answer": tok.decode(out[0, len(ids):].tolist(), skip_special_tokens=True)})
    (SFT / "chat-probes.json").write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
    log("all done: see workspace/run-english-sft/code-eval.json and chat-probes.json")


if __name__ == "__main__":
    main()
