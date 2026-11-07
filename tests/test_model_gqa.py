import unittest

import torch

from localbrain.config import Config
from localbrain.model import LanguageModel


def small(**kw):
    base = dict(arch="v2", width=64, layers=2, heads=8, context=64, vocab_size=300, dropout=0.0)
    return Config(**{**base, **kw})


class GQATests(unittest.TestCase):
    def test_config_validation(self):
        with self.assertRaises(ValueError):
            small(kv_heads=3)  # 8 heads not divisible by 3
        with self.assertRaises(ValueError):
            Config(arch="gpt", width=64, heads=8, kv_heads=2)
        self.assertEqual(small(precision="bf16").precision, "bf16")

    def test_no_grouping_keeps_old_layout(self):
        torch.manual_seed(0)
        old = LanguageModel(small())             # kv_heads default 0 = previous behaviour
        same = LanguageModel(small(kv_heads=8))  # explicit kv_heads == heads
        self.assertEqual(old.blocks[0].qkv.weight.shape, (3 * 64, 64))
        same.load_state_dict(old.state_dict())   # old checkpoints load unchanged
        ids = torch.randint(0, 300, (2, 10))
        self.assertTrue(torch.allclose(old(ids)[0], same(ids)[0], atol=1e-6))

    def test_gqa_smaller_and_trains(self):
        torch.manual_seed(0)
        gqa = LanguageModel(small(kv_heads=2))
        full = LanguageModel(small())
        self.assertEqual(gqa.blocks[0].qkv.weight.shape, (64 + 2 * 2 * 8, 64))
        self.assertLess(sum(p.numel() for p in gqa.parameters()), sum(p.numel() for p in full.parameters()))
        ids = torch.randint(0, 300, (4, 16))
        opt = torch.optim.AdamW(gqa.parameters(), lr=1e-2)
        first = None
        for _ in range(30):
            _, loss = gqa(ids[:, :-1], ids[:, 1:])
            first = first if first is not None else loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), first * 0.5)

    def test_gqa_cached_generation_matches_uncached(self):
        torch.manual_seed(0)
        model = LanguageModel(small(kv_heads=2)).eval()
        prompt = torch.randint(0, 300, (1, 5))
        cached = model.generate(prompt, 12, temperature=1.0, top_k=1)   # uses the KV cache
        ids = prompt
        with torch.no_grad():
            for _ in range(12):                                        # plain full recompute
                nxt = model(ids)[0][:, -1].argmax(-1, keepdim=True)
                ids = torch.cat((ids, nxt), dim=1)
        self.assertTrue(torch.equal(cached, ids))


if __name__ == "__main__":
    unittest.main()
