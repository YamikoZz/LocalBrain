"""English SFT mix: human-written conversations + coding skills, all as question/answer pairs.

Sources
  * OpenAssistant oasst2 (Apache-2.0): human-written prompts and replies, English,
    not deleted, passing review, best-ranked reply (rank 0), synthetic messages excluded.
  * Databricks Dolly 15k (CC-BY-SA-3.0): human-written instruction/response pairs.
  * Generated coding skills (tasks.py): write code with a short sentence before it,
    explain code in sentences, and predict what a function returns.
  * A few hand-written self-introduction/small-talk replies.
"""
import json
import random
from pathlib import Path

from .tasks import F, HELD_OUT, LANGUAGES, literal, make_task

FENCE = {"python": "python", "javascript": "javascript"}

INTRO_QUESTIONS = ["Hi", "Hello", "Hey", "Hello!", "Hi there", "Good morning", "Who are you?",
                   "What are you?", "What can you do?", "Introduce yourself.", "What is your name?",
                   "Can you help me with code?", "How can you help me?", "Are you an AI?"]
INTRO_ANSWERS = [
    "Hello! I'm LocalBrain, a small AI model trained from scratch on a home computer. "
    "I can write short Python and JavaScript functions and explain simple code. "
    "I'm still small, so please test the code I write.",
    "Hi! I'm LocalBrain, a small coding assistant that runs on your own computer. "
    "Describe a function you need, and I'll try to write it in Python or JavaScript.",
    "I'm LocalBrain, a small language model trained at home. I'm best at short coding tasks, "
    "like writing a function from a description or explaining what a piece of code does.",
]


def oasst_pairs(paths):
    import pyarrow.parquet as pq
    rows = []
    for path in paths:
        rows += pq.read_table(path, columns=["message_id", "parent_id", "text", "role", "lang", "deleted",
                                             "review_result", "rank", "synthetic"]).to_pylist()
    by_id = {r["message_id"]: r for r in rows}
    for r in rows:
        if (r["role"] != "assistant" or r["lang"] != "en" or r["deleted"] or r["synthetic"]
                or r["review_result"] is False or r["rank"] not in (0, None)):
            continue
        parent = by_id.get(r["parent_id"])
        # First turn only: the prompt must be a root message so the pair stands alone.
        if not parent or parent["parent_id"] is not None or parent["lang"] != "en" or parent["deleted"]:
            continue
        yield {"question": parent["text"].strip(), "answer": r["text"].strip(), "source": "oasst2"}


def dolly_pairs(path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        question = row["instruction"].strip()
        if row.get("context", "").strip():
            question += "\n\n" + row["context"].strip()
        yield {"question": question, "answer": row["response"].strip(), "source": "dolly"}


def code_skills(count, seed, languages=("python", "javascript")):
    """Coding pairs that also teach sentences: write, explain, and predict output."""
    rng = random.Random(seed)
    pool = [f for f in F if f["name"] not in HELD_OUT]
    leads = ["Here is a {L} function that {d}:", "Sure! This {L} function {d}:",
             "You can do it like this. The function {d}:", "Here's one way to write it in {L}:"]
    for _ in range(count):
        fam, language = rng.choice(pool), rng.choice(languages)
        task = make_task(fam, language, rng)
        code = rng.choice(task["solutions"]).rstrip()
        name, lang = task["function"], LANGUAGES[language]
        desc = task["prompt"].split("` that ", 1)[1].split(".\nExample:")[0]
        example = next((t for t in task["tests"] if t["args"] not in ([[]], [""])), task["tests"][0])
        call = f"{name}({', '.join(literal(a, language) for a in example['args'])})"
        result = literal(example["expected"], language)
        kind = rng.random()
        if kind < 0.6:
            if rng.random() < 0.5:
                question, answer = task["prompt"], code + "\n"          # code only
            else:
                lead = rng.choice(leads).format(L=lang, d=desc)
                question = task["prompt"]
                answer = f"{lead}\n\n```{FENCE[language]}\n{code}\n```\n\nFor example, {call} returns {result}."
        elif kind < 0.8:
            question = rng.choice(["Explain what this {L} code does:", "What does this {L} function do?",
                                   "Can you explain this code?"]).format(L=lang)
            question += f"\n\n```{FENCE[language]}\n{code}\n```"
            answer = f"The function `{name}` {desc}. For example, {call} returns {result}."
        else:
            question = f"What does {call} return for this {lang} code?\n\n```{FENCE[language]}\n{code}\n```"
            answer = f"It returns {result}."
        yield {"question": question, "answer": answer, "source": "code-" + fam["name"], "family": fam["name"]}


def build_mix(oasst_files, dolly_file, output, code_count=70000, seed=11):
    output = Path(output)
    if output.exists():
        raise ValueError("Output exists; choose a new file")
    rows = list(oasst_pairs(oasst_files)) + list(dolly_pairs(dolly_file))
    rows += list(code_skills(code_count, seed))
    rng = random.Random(seed)
    for _ in range(300):
        rows.append({"question": rng.choice(INTRO_QUESTIONS), "answer": rng.choice(INTRO_ANSWERS),
                     "source": "intro"})
    rng.shuffle(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            key = row["source"].split("-")[0]
            counts[key] = counts.get(key, 0) + 1
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"output": str(output), "rows": len(rows), "by_source": counts}
