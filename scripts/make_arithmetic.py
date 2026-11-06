"""Generate verified arithmetic + operator-recognition SFT pairs.

Both teachers (this one and the external OpenCode agent) found the same gap: the model
has never seen what +, -, *, / mean, so it produces operator word-salad. A 20M model CAN
learn small-number arithmetic if shown enough correct examples. Every answer here is
computed in Python, so the data is 100% verified.

Scope kept to the learnable range (small operands, exact division). Output is the standard
SFT {question, answer, source} JSONL, ready to fold into the next SFT mix.
"""
import json
import random
from pathlib import Path

OUT = Path("workspace/raw-sft/arithmetic.jsonl")

OPERATOR_FACTS = [
    ("What does the + symbol mean in math?",
     "The + symbol means addition: it adds two numbers together. For example, 2 + 3 = 5."),
    ("What does the - symbol mean in math?",
     "The - symbol means subtraction: it takes one number away from another. For example, 7 - 4 = 3."),
    ("What does the * symbol mean in math?",
     "The * symbol means multiplication: it is repeated addition of a number. For example, 3 * 4 = 12."),
    ("What does the / symbol mean in math?",
     "The / symbol means division: it splits a number into equal parts. For example, 12 / 4 = 3."),
    ("What is the difference between + and *?",
     "The + symbol adds two numbers, like 2 + 3 = 5. The * symbol multiplies them, like 2 * 3 = 6."),
    ("What is the difference between - and /?",
     "The - symbol subtracts one number from another, like 8 - 3 = 5. The / symbol divides, like 8 / 2 = 4."),
]

ADD_PHRASES = ["What is {a} + {b}?", "Calculate {a} plus {b}.", "{a} + {b} = ?", "Add {a} and {b}."]
SUB_PHRASES = ["What is {a} - {b}?", "Calculate {a} minus {b}.", "{a} - {b} = ?", "Subtract {b} from {a}."]
MUL_PHRASES = ["What is {a} * {b}?", "Calculate {a} times {b}.", "{a} * {b} = ?", "Multiply {a} by {b}."]
DIV_PHRASES = ["What is {a} / {b}?", "Calculate {a} divided by {b}.", "{a} / {b} = ?", "Divide {a} by {b}."]


def pair(q, a):
    return {"question": q, "answer": a, "source": "arithmetic"}


def generate(seed=7):
    rng = random.Random(seed)
    rows, seen = [], set()

    def add(q, a):
        if q not in seen:
            seen.add(q)
            rows.append(pair(q, a))

    for q, a in OPERATOR_FACTS:
        add(q, a)

    # Addition and subtraction over 0..50 (subtraction kept non-negative).
    for a in range(0, 51):
        for b in range(0, 51):
            add(rng.choice(ADD_PHRASES).format(a=a, b=b), f"{a} + {b} = {a + b}")
            if a >= b:
                add(rng.choice(SUB_PHRASES).format(a=a, b=b), f"{a} - {b} = {a - b}")

    # Multiplication tables 0..12.
    for a in range(0, 13):
        for b in range(0, 13):
            add(rng.choice(MUL_PHRASES).format(a=a, b=b), f"{a} * {b} = {a * b}")

    # Division: only exact results (a = b * c), divisor 1..12.
    for b in range(1, 13):
        for c in range(0, 13):
            a = b * c
            add(rng.choice(DIV_PHRASES).format(a=a, b=b), f"{a} / {b} = {c}")

    rng.shuffle(rows)
    return rows


def main():
    rows = generate()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(OUT)}))


if __name__ == "__main__":
    main()
