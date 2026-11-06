"""Interactive window for trying a checkpoint. Press Enter to send.

Base model (pretraining): it continues whatever you type (best with the start of some code).
Instruction model (after SFT): it answers your message.
"""
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


def read_message():
    """One line = one message. End a line with a backslash to continue on the next line."""
    lines = []
    while True:
        line = input("... " if lines else "You> ")
        if line.endswith("\\"):
            lines.append(line[:-1])
            continue
        lines.append(line)
        text = "\n".join(lines)
        if text.strip():
            return text
        lines = []


def main(run, device="auto", max_tokens=160, temperature=0.6):
    model, tok, step, kind = load(run, device)
    device = next(model.parameters()).device
    instruct = kind == "instruction"
    print(f"LocalBrain playground | {run} | step {step} | "
          + ("chat model: type a message" if instruct
             else "base model (not a chat model yet): it continues the text you type"))
    print("Press Enter to send. End a line with \\ to type more lines. Commands: /reload, /exit\n")
    while True:
        try:
            text = read_message()
        except EOFError:
            return
        command = text.strip()
        if command in {"/exit", "/quit"}:
            return
        if command == "/reload":
            model, tok, step, kind = load(run, str(device))
            instruct = kind == "instruction"
            print(f"reloaded step {step}\n")
            continue
        print("(thinking...)", flush=True)
        if instruct:
            ids = [tok.token_to_id("[BOS]")] + tok.encode(prompt_text(text)).ids
        else:
            ids = tok.encode(text).ids
        ids = ids[-(model.cfg.context - max_tokens):]
        out = model.generate(torch.tensor([ids], device=device), max_tokens, temperature=temperature,
                             top_k=40, eos_id=tok.token_to_id("[EOS]"),
                             repetition_penalty=1.3, no_repeat_ngram=3)
        reply = tok.decode(out[0, len(ids):].tolist(), skip_special_tokens=True)
        print("Model> " + (reply.strip() if instruct else text + reply) + "\n" + "-" * 60)
