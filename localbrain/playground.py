"""Interactive window for trying a checkpoint: type the start of some code, it continues."""
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .config import Config
from .instruction import prompt_text
from .model import LanguageModel
from .train import choose_device


def load(run, device="auto"):
    run = Path(run)
    path = run / "best.pt" if (run / "best.pt").exists() else run / "latest.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    cfg = Config(**saved["config"])
    model = LanguageModel(cfg)
    model.load_state_dict(saved["model"])
    model.to(choose_device(device)).eval()
    return model, Tokenizer.from_file(str(run / "tokenizer.json")), saved["step"], saved.get("training_kind")


def main(run, device="auto", max_tokens=120, temperature=0.6):
    model, tok, step, kind = load(run, device)
    instruct = kind == "instruction"
    device = next(model.parameters()).device
    print(f"LocalBrain playground | {run} | step {step} | "
          + ("instruction model: describe the function you want" if instruct
             else "base model: type the START of some code and it will continue"))
    print("Finish your input with an empty line. Type /exit to quit, /reload for the newest checkpoint.\n")
    while True:
        lines = []
        try:
            while True:
                line = input("... " if lines else ">>> ")
                if not line and lines:
                    break
                if line.strip() in {"/exit", "/quit"}:
                    return
                if line.strip() == "/reload":
                    model, tok, step, kind = load(run, str(device))
                    instruct = kind == "instruction"
                    print(f"reloaded step {step}")
                    lines = []
                    continue
                if line or lines:
                    lines.append(line)
        except EOFError:
            return
        text = "\n".join(lines)
        ids = ([tok.token_to_id("[BOS]")] + tok.encode(prompt_text(text)).ids) if instruct else tok.encode(text).ids
        ids = ids[-(model.cfg.context - max_tokens):]
        out = model.generate(torch.tensor([ids], device=device), max_tokens, temperature=temperature,
                             top_k=40, eos_id=tok.token_to_id("[EOS]"), repetition_penalty=1.1)
        continuation = tok.decode(out[0, len(ids):].tolist(), skip_special_tokens=True)
        print("\n" + ("" if instruct else text) + continuation + "\n" + "-" * 60)
