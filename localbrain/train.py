import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .config import Config
from .data import TokenData, fingerprint
from .model import LanguageModel
from .optim import CPUAdamW, Combined, Muon


def choose_device(request):
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if request == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; install a CUDA PyTorch build or select cpu")
    return torch.device(request)


def checkpoint_save(path, model, optimizer, step, cfg, data, elapsed):
    state = {
        "format_version": 1, "config": cfg.as_dict(), "step": step,
        "training_kind": data.meta.get("kind", "language"),
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "tokenizer_sha256": data.meta["tokenizer_sha256"],
        "dataset_sha256": fingerprint(data.folder / "meta.json"),
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(), "data_rng": data.rng.bit_generator.state,
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "elapsed_seconds": elapsed,
    }
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def build_optimizer(model, cfg, device):
    if cfg.optimizer == "cpu_adamw":
        return CPUAdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    fused = {"fused": True} if device.type == "cuda" else {}
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                 weight_decay=cfg.weight_decay, **fused)
    # Muon for 2-D matrices inside blocks; AdamW for embeddings/head (tied) and norms.
    hidden = [p for n, p in model.named_parameters() if n.startswith("blocks.") and p.ndim == 2]
    ids = {id(p) for p in hidden}
    rest = [p for p in model.parameters() if id(p) not in ids]
    return Combined([Muon(hidden, lr=cfg.muon_lr, weight_decay=cfg.weight_decay),
                     torch.optim.AdamW(rest, lr=cfg.learning_rate, betas=(0.9, 0.95),
                                       weight_decay=0.0, **fused)])


def lr_scale(step, cfg):
    """Warmup-stable-decay: linear warmup, flat, then linear decay over the final fraction."""
    if cfg.schedule == "constant":
        return 1.0
    if cfg.warmup_steps and step <= cfg.warmup_steps:
        return step / cfg.warmup_steps
    decay_start = cfg.steps * (1 - cfg.decay_fraction)
    if step <= decay_start:
        return 1.0
    progress = (step - decay_start) / max(1, cfg.steps - decay_start)
    return 1.0 - (1.0 - cfg.min_lr_ratio) * min(1.0, progress)


def load_weights(path, model, data):
    """Start a new run from another run's weights (fresh optimizer), e.g. pretrain -> SFT."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["tokenizer_sha256"] != data.meta["tokenizer_sha256"]:
        raise ValueError("--init checkpoint tokenizer differs from this dataset's tokenizer")
    defaults = {"arch": "gpt", "kv_heads": 0}
    for key in ("arch", "layers", "width", "heads", "kv_heads", "vocab_size"):
        if saved["config"].get(key, defaults.get(key)) != getattr(model.cfg, key):
            raise ValueError(f"--init checkpoint differs in {key}")
    model.load_state_dict(saved["model"])
    return saved["step"]


def autocast(device, cfg):
    dtype = torch.bfloat16 if cfg.precision == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype,
                          enabled=cfg.precision in {"fp16", "bf16"} and device.type == "cuda")


@torch.no_grad()
def evaluate(model, data, batches, device):
    model.eval()
    rng_state = data.rng.bit_generator.state
    values = []
    data.rng.bit_generator.state = type(data.rng.bit_generator)(12345).state
    try:
        for _ in range(batches):
            x, y = data.batch("val", device)
            with autocast(device, model.cfg):
                _, loss = model(x, y)
            values.append(loss.item())
    finally:
        data.rng.bit_generator.state = rng_state
        model.train()
    return sum(values) / len(values)


def train(dataset, run, cfg, resume=False, init=None):
    if cfg.polite:
        from .hardware import be_polite
        be_polite()
    torch.set_num_threads(cfg.cpu_threads)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = choose_device(cfg.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True  # Ampere+ (L4/A100); ignored on T4
        torch.backends.cudnn.allow_tf32 = True
    if device.type == "cuda" and cfg.vram_limit_gib:
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, cfg.vram_limit_gib / total))
    kind = json.loads((Path(dataset) / "meta.json").read_text(encoding="utf-8")).get("kind")
    if kind == "instruction":
        from .instruction import InstructionData
        data = InstructionData(dataset, cfg.context, cfg.batch_size, cfg.seed)
        data.curriculum = cfg.curriculum
    else:
        data = TokenData(dataset, cfg.context, cfg.batch_size, cfg.seed)
    cfg.vocab_size = data.meta["vocab_size"]
    run = Path(run)
    checkpoint_path = run / "latest.pt"
    if checkpoint_path.exists() and not resume:
        raise ValueError("Run already contains a model; use --resume or a new --run")
    if resume and not checkpoint_path.is_file():
        raise ValueError("No checkpoint to resume")
    saved = None
    if resume:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        for key in cfg.as_dict():
            default = Config().as_dict()[key]
            if key not in {"steps", "device", "cpu_threads", "eval_every", "eval_batches",
                           "save_every", "polite", "vram_limit_gib", "decay_fraction", "compile",
                           "min_lr_ratio"} and cfg.as_dict()[key] != saved["config"].get(key, default):
                raise ValueError(f"Resume config differs: {key}")
        if saved["tokenizer_sha256"] != data.meta["tokenizer_sha256"]:
            raise ValueError("Checkpoint tokenizer does not match dataset")
        if saved["dataset_sha256"] != fingerprint(Path(dataset) / "meta.json"):
            raise ValueError("Resume dataset differs")
    if init and resume:
        raise ValueError("Use either --init (new run) or --resume, not both")
    model = LanguageModel(cfg)
    init_step = load_weights(init, model, data) if init else None
    model.to(device)
    # Compiled wrapper is used only for the training forward; checkpoints save the plain model.
    forward = torch.compile(model) if cfg.compile and device.type == "cuda" else model
    optimizer = build_optimizer(model, cfg, device)
    for group in optimizer.param_groups:
        group["base_lr"] = group["lr"]
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.precision == "fp16" and device.type == "cuda")
    start = 0
    previous_elapsed = 0.0
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["step"]
        previous_elapsed = saved["elapsed_seconds"]
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        data.rng.bit_generator.state = saved["data_rng"]
        if device.type == "cuda" and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        del saved
    if cfg.steps <= start:
        raise ValueError(f"steps must exceed checkpoint step {start}")
    run.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(dataset) / "tokenizer.json", run / "tokenizer.json")
    (run / "config.json").write_text(json.dumps(cfg.as_dict(), indent=2), encoding="utf-8")
    parameters = sum(p.numel() for p in model.parameters())
    print(json.dumps({"device": str(device), "parameters": parameters, "arch": cfg.arch,
                      "init_from": str(init) if init else None, "init_step": init_step,
                      "optimizer": cfg.optimizer, "precision": cfg.precision, "start_step": start,
                      "target_step": cfg.steps}, ensure_ascii=False), flush=True)
    model.train()
    best_path = run / "best.json"
    best_val = json.loads(best_path.read_text())["val_loss"] if resume and best_path.exists() else float("inf")
    began = time.monotonic()
    step = start
    interrupted = False
    # A signal during an update is not saved as a complete optimizer step.
    try:
        for current in range(start + 1, cfg.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            scale = lr_scale(current, cfg)
            data.progress = current / cfg.steps
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * scale
            total = 0.0
            for _ in range(cfg.accumulation):
                x, y = data.batch("train", device)
                with autocast(device, cfg):
                    _, loss = forward(x, y)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss; reduce learning rate")
                scaler.scale(loss / cfg.accumulation).backward()
                total += loss.detach().item() / cfg.accumulation
            subs = getattr(optimizer, "optimizers", [optimizer])
            if scaler.is_enabled():
                for sub in subs:
                    scaler.unscale_(sub)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler.is_enabled():
                for sub in subs:
                    scaler.step(sub)
                scaler.update()
            else:
                if not torch.isfinite(norm):
                    raise RuntimeError("Non-finite gradient; latest checkpoint preserved")
                optimizer.step()
            step = current
            report = {"step": step, "train_loss": round(total, 5), "lr_scale": round(scale, 4),
                      "elapsed_seconds": round(time.monotonic() - began, 2)}
            if step == start + 1 or step % cfg.eval_every == 0 or step == cfg.steps:
                report["val_loss"] = round(evaluate(model, data, cfg.eval_batches, device), 5)
                if device.type == "cuda":
                    report["peak_vram_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
                with (run / "metrics.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(report) + "\n")
                print(json.dumps(report), flush=True)
                if kind == "instruction" and report["val_loss"] < best_val:
                    best_val = report["val_loss"]
                    checkpoint_save(run / "best.pt", model, optimizer, step, cfg, data,
                                    previous_elapsed + time.monotonic() - began)
                    best_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            if step % cfg.save_every == 0 or step == cfg.steps:
                checkpoint_save(checkpoint_path, model, optimizer, step, cfg, data,
                                previous_elapsed + time.monotonic() - began)
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted. Resume from the last fully saved checkpoint.", flush=True)
    return {"step": step, "interrupted": interrupted, "checkpoint": str(checkpoint_path),
            "elapsed_seconds": round(time.monotonic() - began, 2)}


def generate(run, prompt, max_tokens=100, temperature=0.8, device="auto"):
    run = Path(run)
    saved = torch.load(run / "latest.pt", map_location="cpu", weights_only=True)
    if fingerprint(run / "tokenizer.json") != saved["tokenizer_sha256"]:
        raise ValueError("Run tokenizer does not match checkpoint")
    cfg = Config(**saved["config"])
    torch.set_num_threads(cfg.cpu_threads)
    target = choose_device(device)
    model = LanguageModel(cfg)
    model.load_state_dict(saved["model"])
    del saved
    model.to(target)
    tokenizer = Tokenizer.from_file(str(run / "tokenizer.json"))
    ids = tokenizer.encode(prompt).ids or [tokenizer.token_to_id("[BOS]")]
    # Keep only the supported prompt tail; generation uses a sliding window.
    ids = ids[-cfg.context:]
    x = torch.tensor([ids], dtype=torch.long, device=target)
    out = model.generate(x, max_tokens, temperature,
                         eos_id=tokenizer.token_to_id("[EOS]"))
    return tokenizer.decode(out[0, len(ids):].tolist(), skip_special_tokens=True)
