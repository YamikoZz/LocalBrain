import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Config:
    layers: int = 6
    width: int = 256
    heads: int = 8
    context: int = 256
    vocab_size: int = 8192
    dropout: float = 0.1
    batch_size: int = 1
    accumulation: int = 8
    steps: int = 1000
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    device: str = "auto"
    optimizer: str = "cpu_adamw"
    checkpoint_blocks: bool = True
    cpu_threads: int = 8
    eval_every: int = 100
    eval_batches: int = 8
    save_every: int = 100
    seed: int = 42
    # Fields below default to the original behaviour so older configs/checkpoints load unchanged.
    arch: str = "gpt"
    muon_lr: float = 0.02
    schedule: str = "constant"
    warmup_steps: int = 0
    decay_fraction: float = 0.2
    min_lr_ratio: float = 0.1
    precision: str = "fp32"
    vram_limit_gib: float = 0.0
    polite: bool = False
    curriculum: bool = False  # instruction data: start with short examples, widen to all
    kv_heads: int = 0          # grouped-query attention: key/value heads (0 = same as heads)
    compile: bool = False      # torch.compile the training forward pass (CUDA)

    def __post_init__(self):
        for key in ("layers", "width", "heads", "context", "vocab_size",
                    "batch_size", "accumulation", "steps", "cpu_threads",
                    "eval_every", "eval_batches", "save_every"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if self.kv_heads < 0 or (self.kv_heads and self.heads % self.kv_heads):
            raise ValueError("kv_heads must divide heads (0 = no grouping)")
        if self.kv_heads and self.arch != "v2":
            raise ValueError("kv_heads needs arch v2")
        if self.vocab_size < 260 or self.vocab_size > 65535:
            raise ValueError("vocab_size must be between 260 and 65535")
        if self.context < 2 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid context or dropout")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")
        if self.optimizer not in {"adamw", "cpu_adamw", "muon"}:
            raise ValueError("optimizer must be adamw, cpu_adamw or muon")
        if self.arch not in {"gpt", "v2"}:
            raise ValueError("arch must be gpt or v2")
        if self.arch == "v2" and (self.width // self.heads) % 2:
            raise ValueError("v2 needs an even head dimension for rotary embeddings")
        if self.schedule not in {"constant", "wsd"}:
            raise ValueError("schedule must be constant or wsd")
        if self.warmup_steps < 0 or not 0 <= self.decay_fraction <= 1 or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("Invalid warmup_steps, decay_fraction or min_lr_ratio")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16 or bf16")
        if self.vram_limit_gib < 0 or self.muon_lr <= 0:
            raise ValueError("Invalid vram_limit_gib or muon_lr")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid learning_rate or weight_decay")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def as_dict(self):
        return asdict(self)
