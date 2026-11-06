"""Import a small, balanced, verified sample of "LocalBrain Programming Dataset v4".

v4 has 2.6M rows but only ~38 task families (28 one-line arithmetic-expression functions,
~10 SQL patterns) repeated with different constants and names, in Python, JavaScript,
TypeScript, Luau, C++ and SQL. Its own README warns that training on all of it degrades a
model. What it adds for us is function SYNTAX in languages our model confuses (Luau, TS, C++),
plus "find the bug" and "predict the return value" exercises.

So we take a capped random sample per (language, family, kind) and verify what we can,
WITHOUT executing any code from the dataset (it is external, untrusted input):
  * Python: every response must parse (ast.parse, no exec); execution traces are re-computed by
    a whitelist evaluator (+ - * / // % ** abs min max, comparisons, if-else) and must match;
  * execution traces in other languages cannot be checked here, so they are dropped;
  * TypeScript / C++ / Luau / SQL / JS code answers are kept as the dataset's templates.
The dataset's own factory/ scripts are NOT run; this only reads the gzip JSONL shards.
Output: workspace/raw-sft/v4-sample.jsonl in the standard {question, answer, source} format.
"""
import argparse
import ast
import collections
import glob
import gzip
import json
import operator
import os
import random

CAPS = {"instruction_sft": 40, "debugging": 12, "execution_trace": 8}


def sample(root, seed):
    rng = random.Random(seed)
    seen = collections.Counter()
    keep = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(root, "dataset", "train", "*", "*.jsonl.gz"))):
        with gzip.open(path, "rt", encoding="utf-8") as h:
            for line in h:
                r = json.loads(line)
                key = (r["language"], r["family"], r["kind"])
                cap = CAPS.get(r["kind"], 0)
                seen[key] += 1
                if len(keep[key]) < cap:
                    keep[key].append(r)
                else:  # reservoir sampling so every row has the same chance
                    j = rng.randrange(seen[key])
                    if j < cap:
                        keep[key][j] = r
    return [r for rows in keep.values() for r in rows]


BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
       ast.Gt: operator.gt, ast.GtE: operator.ge}
FUNCS = {"abs": abs, "min": min, "max": max}


class Unsupported(Exception):
    pass


def evaluate(node, env):
    """Whitelist arithmetic evaluator: never runs dataset code, only interprets these node types."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
        return node.value
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Not)):
        v = evaluate(node.operand, env)
        return -v if isinstance(node.op, ast.USub) else (+v if isinstance(node.op, ast.UAdd) else not v)
    if isinstance(node, ast.BinOp) and type(node.op) in BIN:
        left, right = evaluate(node.left, env), evaluate(node.right, env)
        if isinstance(node.op, ast.Pow) and abs(right) > 8:
            raise Unsupported("large power")
        return BIN[type(node.op)](left, right)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in CMP:
        return CMP[type(node.ops[0])](evaluate(node.left, env), evaluate(node.comparators[0], env))
    if isinstance(node, ast.IfExp):
        return evaluate(node.body if evaluate(node.test, env) else node.orelse, env)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCS
            and not node.keywords):
        return FUNCS[node.func.id](*(evaluate(a, env) for a in node.args))
    raise Unsupported(type(node).__name__)


def trace_parts(r):
    body, call = r["instruction"].split("\n\n", 1)[1].rsplit("\n\nCall: ", 1)
    return body, call.strip()


def check_python(r):
    try:
        if r["kind"] != "execution_trace":
            ast.parse(r["response"])  # syntax only, nothing is executed
            return True
        body, call = trace_parts(r)
        fn = ast.parse(body).body
        if len(fn) != 1 or not isinstance(fn[0], ast.FunctionDef) or len(fn[0].body) != 1 \
                or not isinstance(fn[0].body[0], ast.Return):
            return False
        c = ast.parse(call, mode="eval").body
        if not isinstance(c, ast.Call) or len(c.args) != len(fn[0].args.args):
            return False
        env = {p.arg: evaluate(a, {}) for p, a in zip(fn[0].args.args, c.args)}
        return str(evaluate(fn[0].body[0].value, env)) == r["response"].strip()
    except (SyntaxError, Unsupported, ArithmeticError, ValueError, TypeError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="LocalBrain_Programming_Dataset_v4")
    ap.add_argument("--output", default="workspace/raw-sft/v4-sample.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = sample(args.root, args.seed)
    stats = collections.Counter()
    out = []
    for r in rows:
        lang, kind = r["language"], r["kind"]
        if lang == "python":
            ok = check_python(r)
        elif kind == "execution_trace":
            ok = False  # cannot verify the expected value without running foreign code
        else:
            ok = True  # code answer kept as the dataset's template
        stats[f"{lang}/{'ok' if ok else 'FAIL'}"] += 1
        if ok:
            out.append({"question": r["instruction"], "answer": r["response"].rstrip() + "\n",
                        "source": f"v4-{lang}-{kind}"})
    random.Random(args.seed).shuffle(out)
    with open(args.output, "w", encoding="utf-8") as h:
        for row in out:
            h.write(json.dumps(row, ensure_ascii=False) + "\n")
    by_lang = collections.Counter(r["source"].split("-")[1] for r in out)
    by_kind = collections.Counter(r["source"].split("-", 2)[2] for r in out)
    print(json.dumps({"rows": len(out), "by_language": by_lang, "by_kind": by_kind,
                      "verification": dict(sorted(stats.items())), "output": args.output}, indent=1))


if __name__ == "__main__":
    main()
