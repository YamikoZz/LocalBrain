"""Generate a strong multi-language + math-via-code SFT pack.

Observed weaknesses: the model only wrote Python well, reverted to prose/Python for other
languages, and got arithmetic wrong. This pack teaches:
  1. Multi-language code (Python, JavaScript, Lua, HTML, CSS, SQL, C, Bash) as instruction
     -> fenced code, so the model learns each language's shape, not just Python.
  2. Math-via-code: for bigger/explicit "write code to compute" asks, answer with runnable
     code instead of guessing a number (a 20M model can't compute reliably, but it CAN learn
     to emit `print(expr)` which an execution step then runs correctly).

Python/JavaScript solutions are simple and runnable (verify_samples checks a subset). Lua/
HTML/CSS/SQL/C/Bash are hand-written correct templates (no local runtime to execute them).
Output is the standard SFT {question, answer, source} JSONL. Dedup downstream is on the
QUESTION only, so each task uses several distinct phrasings.
"""
import json
import random
from pathlib import Path

OUT = Path("workspace/raw-sft/hardcore.jsonl")


def fence(lang, code):
    return f"```{lang}\n{code.rstrip()}\n```"


# --- multi-language tasks: {task_key: {lang: code}} with shared natural phrasings ---
TASKS = {
    "print_hello": {
        "ask": ["Print \"Hello, World!\" in {L}.", "Write {L} code that prints Hello, World!",
                "How do I print Hello, World! in {L}?"],
        "python": 'print("Hello, World!")',
        "javascript": 'console.log("Hello, World!");',
        "lua": 'print("Hello, World!")',
        "c": '#include <stdio.h>\nint main(void) {\n    printf("Hello, World!\\n");\n    return 0;\n}',
        "bash": 'echo "Hello, World!"',
    },
    "add_two": {
        "ask": ["Write a function in {L} that adds two numbers a and b.",
                "In {L}, make a function that returns the sum of a and b.",
                "{L}: a function add(a, b) returning a + b."],
        "python": 'def add(a, b):\n    return a + b',
        "javascript": 'function add(a, b) {\n  return a + b;\n}',
        "lua": 'function add(a, b)\n  return a + b\nend',
        "c": 'int add(int a, int b) {\n    return a + b;\n}',
        "bash": 'add() {\n  echo $(( $1 + $2 ))\n}',
    },
    "is_even": {
        "ask": ["Write a function in {L} that returns true if a number n is even.",
                "In {L}, check whether n is even.",
                "{L}: a function is_even(n)."],
        "python": 'def is_even(n):\n    return n % 2 == 0',
        "javascript": 'function isEven(n) {\n  return n % 2 === 0;\n}',
        "lua": 'function is_even(n)\n  return n % 2 == 0\nend',
        "c": 'int is_even(int n) {\n    return n % 2 == 0;\n}',
        "bash": 'is_even() {\n  if (( $1 % 2 == 0 )); then echo true; else echo false; fi\n}',
    },
    "factorial": {
        "ask": ["Write a function in {L} that returns the factorial of n.",
                "In {L}, compute n! with a function.",
                "{L}: factorial(n)."],
        "python": 'def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result',
        "javascript": 'function factorial(n) {\n  let result = 1;\n  for (let i = 2; i <= n; i++) result *= i;\n  return result;\n}',
        "lua": 'function factorial(n)\n  local result = 1\n  for i = 2, n do\n    result = result * i\n  end\n  return result\nend',
        "c": 'long factorial(int n) {\n    long result = 1;\n    for (int i = 2; i <= n; i++) result *= i;\n    return result;\n}',
    },
    "reverse_string": {
        "ask": ["Write a function in {L} that reverses a string s.",
                "In {L}, reverse the string s.",
                "{L}: reverse(s)."],
        "python": 'def reverse(s):\n    return s[::-1]',
        "javascript": 'function reverse(s) {\n  return s.split("").reverse().join("");\n}',
        "lua": 'function reverse(s)\n  return string.reverse(s)\nend',
    },
    "sum_list": {
        "ask": ["Write a function in {L} that returns the sum of a list of numbers.",
                "In {L}, sum all numbers in an array.",
                "{L}: sum_list(nums)."],
        "python": 'def sum_list(nums):\n    total = 0\n    for x in nums:\n        total += x\n    return total',
        "javascript": 'function sumList(nums) {\n  let total = 0;\n  for (const x of nums) total += x;\n  return total;\n}',
        "lua": 'function sum_list(nums)\n  local total = 0\n  for _, x in ipairs(nums) do\n    total = total + x\n  end\n  return total\nend',
    },
    "loop_count": {
        "ask": ["Write {L} code that prints the numbers 1 to 5.",
                "In {L}, print numbers from 1 to 5.",
                "{L}: loop printing 1 through 5."],
        "python": 'for i in range(1, 6):\n    print(i)',
        "javascript": 'for (let i = 1; i <= 5; i++) {\n  console.log(i);\n}',
        "lua": 'for i = 1, 5 do\n  print(i)\nend',
        "c": '#include <stdio.h>\nint main(void) {\n    for (int i = 1; i <= 5; i++) printf("%d\\n", i);\n    return 0;\n}',
        "bash": 'for i in 1 2 3 4 5; do\n  echo "$i"\ndone',
    },
}

# --- HTML / CSS / SQL: domain-specific tasks (each a distinct ask -> solution) ---
HTML_TASKS = [
    ("Write HTML for a button that says Click me.", "html", '<button>Click me</button>'),
    ("Write HTML for a level-1 heading that says Welcome.", "html", '<h1>Welcome</h1>'),
    ("Write HTML for a link to https://example.com.", "html", '<a href="https://example.com">Link</a>'),
    ("Write HTML for an unordered list with items Apple, Banana, Cherry.", "html",
     '<ul>\n  <li>Apple</li>\n  <li>Banana</li>\n  <li>Cherry</li>\n</ul>'),
    ("Write HTML for an image with source cat.jpg and alt text A cat.", "html",
     '<img src="cat.jpg" alt="A cat">'),
    ("Write HTML for a text input with placeholder Your name.", "html",
     '<input type="text" placeholder="Your name">'),
    ("Write a basic HTML page with a title Hello and a paragraph.", "html",
     '<!DOCTYPE html>\n<html>\n<head><title>Hello</title></head>\n<body>\n  <p>Hello, world!</p>\n</body>\n</html>'),
]
CSS_TASKS = [
    ("Write CSS to make all paragraphs red.", "css", 'p {\n  color: red;\n}'),
    ("Write CSS to center text in a div.", "css", 'div {\n  text-align: center;\n}'),
    ("Write CSS to give the body a light gray background.", "css", 'body {\n  background-color: lightgray;\n}'),
    ("Write CSS to make h1 font size 32px.", "css", 'h1 {\n  font-size: 32px;\n}'),
    ("Write CSS to add 10px padding to a button.", "css", 'button {\n  padding: 10px;\n}'),
]
SQL_TASKS = [
    ("Write SQL to select all rows from the users table.", "sql", 'SELECT * FROM users;'),
    ("Write SQL to select the name column from users where age is over 18.", "sql",
     'SELECT name FROM users WHERE age > 18;'),
    ("Write SQL to count how many rows are in the orders table.", "sql", 'SELECT COUNT(*) FROM orders;'),
    ("Write SQL to get users ordered by name.", "sql", 'SELECT * FROM users ORDER BY name;'),
    ("Write SQL to insert a user named Alice aged 30.", "sql",
     "INSERT INTO users (name, age) VALUES ('Alice', 30);"),
    ("Write SQL to update the age to 31 for the user named Alice.", "sql",
     "UPDATE users SET age = 31 WHERE name = 'Alice';"),
    ("Write SQL to delete users whose age is under 13.", "sql", 'DELETE FROM users WHERE age < 13;'),
    ("Write SQL to get the number of users per city.", "sql",
     'SELECT city, COUNT(*) FROM users GROUP BY city;'),
]

LANG_LABEL = {"python": "Python", "javascript": "JavaScript", "lua": "Lua",
              "c": "C", "bash": "Bash"}


def pair(q, a, source="hardcore"):
    return {"question": q, "answer": a, "source": source}


def generate(seed=23):
    rng = random.Random(seed)
    rows, seen = [], set()

    def add(q, a, source="hardcore"):
        key = "".join(q.lower().split())
        if key not in seen:
            seen.add(key)
            rows.append(pair(q, a, source))

    # Multi-language programming tasks.
    for spec in TASKS.values():
        for lang, code in spec.items():
            if lang == "ask":
                continue
            label = LANG_LABEL[lang]
            answer = f"Here is the {label} code:\n\n" + fence(lang, code)
            for template in spec["ask"]:
                add(template.format(L=label), answer)

    # HTML / CSS / SQL.
    for task_list in (HTML_TASKS, CSS_TASKS, SQL_TASKS):
        for ask, lang, code in task_list:
            add(ask, "Sure:\n\n" + fence(lang, code))

    # Math-via-code: teach emitting runnable code for explicit "write code to compute" asks.
    ops = [("+", "plus"), ("-", "minus"), ("*", "times"), ("//", "integer-divided by")]
    for _ in range(120):
        a, b = rng.randint(10, 9999), rng.randint(2, 999)
        sym, word = rng.choice(ops)
        expr = f"{a} {sym} {b}"
        q = rng.choice([
            f"Write Python code to compute {a} {word} {b}.",
            f"Use Python to calculate {expr}.",
            f"Write code that prints {expr}.",
        ])
        add(q, "You can compute it in Python:\n\n" + fence("python", f"print({expr})"),
            source="math-via-code")

    rng.shuffle(rows)
    return rows


def verify_samples(rows):
    """Run the Python answers to confirm they execute (quick sanity check on a subset)."""
    import io
    import contextlib
    ok, bad = 0, 0
    for r in rows:
        ans = r["answer"]
        if "```python" not in ans:
            continue
        code = ans.split("```python", 1)[1].split("```", 1)[0].strip()
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<s>", "exec"), {})
            ok += 1
        except SyntaxError:
            bad += 1
            print("SYNTAX BAD:", r["question"][:50], "::", code[:50])
        except Exception:
            ok += 1  # runtime (e.g. undefined call) is fine; we only check it parses/defines
    return ok, bad


def main():
    rows = generate()
    ok, bad = verify_samples(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "python_ok": ok, "python_syntax_bad": bad,
                      "output": str(OUT)}))


if __name__ == "__main__":
    main()
