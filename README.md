# LocalBrain

**English** · [ภาษาไทย](README.th.md)

**A code model built from scratch by one person on a hobby budget.** Every weight starts from random
initialization. No Llama, Qwen or any other pretrained checkpoint is used, only public datasets.

You describe a function in English. The model writes code, a local runner executes tests, and an agent loop asks
the model to repair failures until the tests pass.

| | |
|---|---|
| Current model | **20.5M parameters**, trained for **≈12 THB (≈$0.35)** of Colab compute |
| Next model | **193M parameters** on 4.0B tokens, training now (≈97 Colab units, ≈$10) |
| Honest scorecard | 66.8% on our in-domain suite, but **0% on held-out task types** (details below) |
| Verified | Pass/fail of all 196 recorded answers re-checked independently, all matching |
| Paper | [Technical report v0.1](docs/paper/LocalBrain-technical-report.md) |

## Why this exists

Most small "local" models are fine-tunes of big-company weights. LocalBrain asks a different question: **how far
can one person get when training everything themselves?** That covers the tokenizer, pretraining, fine-tuning,
evaluation and the agent, on rented GPUs costing a few dollars.

Ground rules:
- **No foreign weights.** Everything is trained from scratch.
- **No refusal behaviour trained in.** The fine-tuning data has no "I can't help with that" examples.
- **Honest numbers.** Bad results are reported as plainly as good ones. Estimates are labelled as estimates.

## Results so far

The 20.5M model (G2) was scored on 196 procedurally generated function-writing tasks: 49 families × 2 tasks ×
Python/JavaScript, 8 hidden tests each. Code is executed; there is no LLM judge.

| Split | Greedy pass@1 | pass@8 |
|---|---|---|
| Task families seen in fine-tuning: Python | 76.8% (63/82) | 81.7% |
| Task families seen in fine-tuning: JavaScript | 82.9% (68/82) | 82.9% |
| **Held-out families (never in fine-tuning)** | **0% (0/32)** | 6.3% (2/32) |
| **Total** | **66.8% (131/196)** | 69.9% |

**What this means:** at 20M parameters the model reproduces patterns it has seen and does not generalize. This
score is model-only. The agent loop was not used.

After an external review we fixed a leak in the benchmark: the prompt example used to be one of the scored tests.
We also added a separate agent-assisted score, which gets feedback from 3 public tests and is scored on all tests.
Section 5 of the [paper](docs/paper/LocalBrain-technical-report.md) has the details.

**For context**, here are approximate published HumanEval scores (not measured by us): Codex-25M ≈3%, Codex-85M
≈8%, CodeGen-350M-mono ≈13%, phi-1-small (350M) ≈45%. The 193M model will be evaluated on HumanEval and MBPP.
Training data is decontaminated against both: the filter catches 164/164 and 500/500 prompts.

## What's in the repo

| Path | What it is |
|---|---|
| `localbrain/model.py` | Decoder-only Transformer: RoPE, RMSNorm, QK-Norm, ReLU² MLP, tied embeddings, GQA |
| `localbrain/train.py`, `optim.py` | Training with resume, Muon optimizer, warmup–stable–decay schedule, bf16, `torch.compile` |
| `localbrain/tasks.py`, `runner.py`, `coder.py` | Verifiable coding tasks, a sandboxed test runner, the sample→test→repair loop, `code-eval` |
| `localbrain/agent.py`, `patch.py`, `projindex.py` | Project agent: index, search, SEARCH/REPLACE patches, run tests, roll back, remember |
| `localbrain/knowledge.py`, `webtools.py` | SQLite FTS5 knowledge base, web search with cache |
| `scripts/retrain.py` | One-command pipeline: pretrain → anneal → SFT → eval (resumable) |
| `scripts/import_code_sft.py` | Imports public code SFT datasets with HumanEval/MBPP decontamination |
| `notebooks/LocalBrain_Colab.ipynb` | Train on Google Colab |

**Not included:** trained weights. They will be released once the quality is worth using. You can train the 20M
model yourself on a free or cheap Colab T4.

## Quick start

```bash
python -m venv .venv
# install a PyTorch build for your GPU/CPU: https://pytorch.org/get-started/locally/
.venv/bin/pip install -e .            # Windows: .venv\Scripts\pip install -e .
python -m unittest discover tests     # 75 tests

# full pipeline (pretrain -> anneal -> SFT -> eval); see notebooks/README.md for Colab
python scripts/retrain.py

# evaluate a run: model-only and agent-assisted scores
python -m localbrain.cli code-eval --run workspace/<run> --agent

# write a function, verified against tests
python -m localbrain.cli code "return True if n is prime" --function is_prime --args n --example '{"args": [7], "expected": true}' --example '{"args": [8], "expected": false}' --run workspace/<run>
```

## Roadmap

| Stage | Size | Status |
|---|---|---|
| G1 | 6.9M, Thai chat | done, failed (incoherent); taught us to pretrain and narrow the domain |
| G2 | 20.5M, code | done; results above |
| **G3** | **193M, code** | **training** on 4.0B tokens; SFT with ≈228K decontaminated examples plus extra Lua/TS/JS |
| Next | ≈700M–1B | depends on G3 results and funding; data quality matters more than size (see phi-1) |

Target languages: Python and Lua first, JavaScript and TypeScript as a pair, less C/C++. Prompts are in English.

## Docs

- [Technical report](docs/paper/LocalBrain-technical-report.md) (English)
- [STATE.md](STATE.md): running log of every run and number (Thai)
- [ROADMAP.md](ROADMAP.md): architecture plan (Thai)
- [Wiki](https://github.com/YamikoZz/LocalBrain/wiki): beginner guide (Thai)

## License

Code: MIT. Datasets keep their own licenses (listed in the report).
