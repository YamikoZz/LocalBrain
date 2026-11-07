"""4x retrain: build a ~4x-larger pretrain corpus (reusing the existing tokenizer so the
anneal/SFT datasets stay valid), train a fresh base on it, then re-anneal and re-SFT.

Why bigger data, not just more steps: the old model saw 131M tokens (0.8 epoch of a 160M
corpus). Looping the same 160M 4x would memorize it. Instead we expand to ~640M tokens
(all 8 code shards + FineWeb-Edu) and train 48k steps = ~393M tokens seen (0.6 epoch),
so every step is fresh signal. SFT is shortened (the first run's val bottomed at step 250
then climbed -- classic small-model overfit).

New outputs use *-big / *2 dirs so the current step-250 model stays as a fallback.
Each stage is resumable: re-running the script skips finished stages and resumes a
half-done pretrain from its latest checkpoint.
"""
import os
import subprocess
import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
LOG = ROOT / "workspace" / "retrain.log"
ENV = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

CODE = [f"workspace/raw-code/code-{i:05d}.parquet" for i in range(8)]
FINEWEB = "workspace/raw-code/fineweb-edu-000_00000.parquet"
TOKENIZER = "workspace/dataset-code/tokenizer.json"

BIG = Path("workspace/dataset-code-big")
PRE = Path("workspace/run-code-pretrain-big")
ANNEAL = Path("workspace/run-code-anneal2")
SFT = Path("workspace/run-english-sft2")

# Pretrain config is overridable so the same pipeline runs on a bigger GPU (e.g. Colab T4,
# 16 GiB) without touching the home-PC default. Set LOCALBRAIN_PRETRAIN_CONFIG to
# configs/code-pretrain-colab.json on Colab; left unset it keeps the 4 GiB home settings.
PRETRAIN_CONFIG = os.environ.get("LOCALBRAIN_PRETRAIN_CONFIG", "configs/code-pretrain-big.json")
# Same for SFT: the home config caps VRAM at 2.8 GiB, which OOMs on Colab's first SFT batch.
SFT_CONFIG = os.environ.get("LOCALBRAIN_SFT_CONFIG", "configs/code-sft.json")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
    with LOG.open("a", encoding="utf-8") as h:
        h.write(line)
    print(line, end="", flush=True)


def run(name, args):
    log(f"run: {name}")
    r = subprocess.run([PY, "-m", "localbrain.cli", *args], cwd=ROOT, env=ENV)
    if r.returncode != 0:
        log(f"FAILED ({r.returncode}) at: {name}; stopping")
        raise SystemExit(1)
    log(f"done: {name}")


def main():
    log("=== 4x retrain pipeline start ===")

    # Stage 0: build the ~640M-token corpus (reuse tokenizer). Skip if already built.
    if (BIG / "meta.json").exists():
        log("skip corpus build (dataset-code-big exists)")
    else:
        run("build 640M corpus (8 code shards + FineWeb-Edu, reuse tokenizer)",
            ["build-corpus", "--code", *CODE, "--english", FINEWEB,
             "--output", str(BIG), "--max-tokens", "640000000", "--tokenizer", TOKENIZER])

    # Stage 1: fresh (or resumed) pretrain on the big corpus.
    log(f"pretrain config: {PRETRAIN_CONFIG}")
    pre_args = ["train", "--data", str(BIG), "--run", str(PRE), "--config", PRETRAIN_CONFIG]
    if (PRE / "latest.pt").exists():
        pre_args.append("--resume")
        log("pretrain: resuming from existing checkpoint")
    run("pretrain 48000 on big corpus", pre_args)

    # Stage 2: anneal from the new base (existing dataset-anneal is tokenizer-compatible).
    if (ANNEAL / "latest.pt").exists():
        log("skip anneal (run-code-anneal2 exists)")
    else:
        run("anneal (fresh, from big pretrain)",
            ["train", "--data", "workspace/dataset-anneal", "--run", str(ANNEAL),
             "--config", "configs/code-anneal.json", "--init", str(PRE / "latest.pt")])

    # Stage 3: short anti-overfit SFT.
    if (SFT / "latest.pt").exists():
        log("skip sft (run-english-sft2 exists)")
    else:
        run("sft (short, anti-overfit; dataset now includes verified arithmetic)",
            ["train", "--data", "workspace/dataset-english-sft", "--run", str(SFT),
             "--config", SFT_CONFIG, "--init", str(ANNEAL / "latest.pt")])

    # Stage 4: evaluate the new model.
    run("code-eval on new SFT",
        ["code-eval", "--run", str(SFT), "--output", str(SFT / "code-eval.json"), "--per-family", "2"])

    log("=== 4x retrain pipeline complete ===")


if __name__ == "__main__":
    main()
