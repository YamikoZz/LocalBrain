import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import Config


class Block(nn.Module):
    """Original GPT-2 style block (arch="gpt"); kept so old checkpoints still load."""
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        self.dropout = cfg.dropout
        self.norm1 = nn.LayerNorm(cfg.width)
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width, bias=False)
        self.proj = nn.Linear(cfg.width, cfg.width, bias=False)
        self.norm2 = nn.LayerNorm(cfg.width)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.width, 4 * cfg.width), nn.GELU(),
            nn.Linear(4 * cfg.width, cfg.width), nn.Dropout(cfg.dropout))
        self.residual_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(
            batch, length, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(batch, length, width)
        x = x + self.residual_dropout(self.proj(out))
        return x + self.mlp(self.norm2(x))


def rope_tables(length, dim, device, base=10000.0, offset=0):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    pos = torch.arange(offset, offset + length, device=device, dtype=torch.float32)
    angles = torch.outer(pos, inv)
    return angles.cos()[None, None], angles.sin()[None, None]


def apply_rope(x, cos, sin):
    a, b = x.float().chunk(2, dim=-1)
    return torch.cat((a * cos - b * sin, a * sin + b * cos), dim=-1).type_as(x)


class BlockV2(nn.Module):
    """RMSNorm, rotary positions, QK-norm, ReLU^2 MLP, no biases (arch="v2")."""
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        self.head_dim = cfg.width // cfg.heads
        self.dropout = cfg.dropout
        self.norm1 = nn.RMSNorm(cfg.width)
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.proj = nn.Linear(cfg.width, cfg.width, bias=False)
        self.norm2 = nn.RMSNorm(cfg.width)
        self.up = nn.Linear(cfg.width, 4 * cfg.width, bias=False)
        self.down = nn.Linear(4 * cfg.width, cfg.width, bias=False)

    def forward(self, x, cos, sin, cache=None):
        batch, length, width = x.shape
        qkv = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = apply_rope(self.q_norm(q), cos, sin), apply_rope(self.k_norm(k), cos, sin)
        if cache is not None:
            if "k" in cache:
                k = torch.cat((cache["k"], k), dim=2)
                v = torch.cat((cache["v"], v), dim=2)
            cache["k"], cache["v"] = k, v
        # With a cache and a single new token, attending to every cached key is causal.
        causal = cache is None or length > 1
        if cache is not None and length > 1 and k.size(2) != length:
            raise ValueError("Multi-token step with a non-empty cache is unsupported")
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(batch, length, width)
        x = x + self.proj(out)
        return x + self.down(F.relu(self.up(self.norm2(x))).square())


class LanguageModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.v2 = cfg.arch == "v2"
        self.tokens = nn.Embedding(cfg.vocab_size, cfg.width)
        if self.v2:
            self.blocks = nn.ModuleList([BlockV2(cfg) for _ in range(cfg.layers)])
            self.norm = nn.RMSNorm(cfg.width)
        else:
            self.positions = nn.Embedding(cfg.context, cfg.width)
            self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
            self.norm = nn.LayerNorm(cfg.width)
        self.head = nn.Linear(cfg.width, cfg.vocab_size, bias=False)
        self.apply(self._init)
        if self.v2:
            for block in self.blocks:
                # Residual branches start at zero: each block begins as the identity.
                nn.init.zeros_(block.proj.weight)
                nn.init.zeros_(block.down.weight)
        self.head.weight = self.tokens.weight

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids, targets=None, caches=None, offset=0):
        if ids.size(1) + offset > self.cfg.context:
            raise ValueError("Input exceeds configured context")
        x = self.tokens(ids)
        if self.v2:
            cos, sin = rope_tables(ids.size(1), self.cfg.width // self.cfg.heads, ids.device, offset=offset)
            for i, block in enumerate(self.blocks):
                cache = None if caches is None else caches[i]
                if self.training and self.cfg.checkpoint_blocks and cache is None:
                    x = checkpoint(block, x, cos, sin, use_reentrant=False)
                else:
                    x = block(x, cos, sin, cache)
        else:
            x = x + self.positions(torch.arange(ids.size(1), device=ids.device))
            for block in self.blocks:
                if self.training and self.cfg.checkpoint_blocks:
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
        logits = self.head(self.norm(x))
        if self.v2:
            logits = 30.0 * torch.tanh(logits.float() / 30.0)  # soft cap keeps training stable
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1))
        return logits, loss

    @staticmethod
    def _filter(logits, ids, prompt_length, temperature, top_k, repetition_penalty, no_repeat_ngram):
        logits = logits / temperature
        if repetition_penalty > 1 or no_repeat_ngram:
            for row in range(ids.size(0)):
                history = ids[row, prompt_length:].tolist()
                if repetition_penalty > 1 and history:
                    seen = list(set(history))
                    scores = logits[row, seen]
                    logits[row, seen] = torch.where(
                        scores > 0, scores / repetition_penalty, scores * repetition_penalty)
                n = no_repeat_ngram
                if n and len(history) >= n - 1:
                    prefix = history[-(n - 1):] if n > 1 else []
                    banned = [history[i + n - 1] for i in range(len(history) - n + 1)
                              if history[i:i + n - 1] == prefix]
                    if banned:
                        logits[row, banned] = -float("inf")
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < values[:, [-1]]] = -float("inf")
        return logits

    @torch.inference_mode()
    def generate(self, ids, max_new_tokens=100, temperature=0.8,
                 top_k=40, eos_id=None, repetition_penalty=1.0, no_repeat_ngram=0):
        """Sample continuations. Rows that emit eos keep producing eos (padding)."""
        if max_new_tokens < 1 or temperature <= 0 or top_k < 1:
            raise ValueError("Invalid generation parameters")
        if repetition_penalty < 1 or no_repeat_ngram < 0:
            raise ValueError("Invalid repetition controls")
        self.eval()
        prompt_length = ids.size(1)
        finished = torch.zeros(ids.size(0), dtype=torch.bool, device=ids.device)
        use_cache = self.v2 and prompt_length + max_new_tokens <= self.cfg.context
        caches = [dict() for _ in self.blocks] if use_cache else None
        logits, _ = self(ids[:, -self.cfg.context:], caches=caches)
        for step in range(max_new_tokens):
            logits = self._filter(logits[:, -1, :].float(), ids, prompt_length, temperature, top_k,
                                  repetition_penalty, no_repeat_ngram)
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            if eos_id is not None:
                nxt[finished] = eos_id
                finished |= nxt.squeeze(1) == eos_id
            ids = torch.cat((ids, nxt), dim=1)
            if (eos_id is not None and finished.all()) or step + 1 == max_new_tokens:
                break
            if use_cache:
                logits, _ = self(nxt, caches=caches, offset=ids.size(1) - 1)
            else:
                logits, _ = self(ids[:, -self.cfg.context:])
        return ids
