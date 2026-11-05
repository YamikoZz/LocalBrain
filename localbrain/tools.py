import ast
import math
import operator
from datetime import datetime
from pathlib import Path

from .documents import read_document
from .knowledge import Knowledge
from .sandbox import run_python
from .webtools import read_webpage, search_web

BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
          ast.Mod: operator.mod, ast.Pow: operator.pow}


def calculate(expression):
    if not isinstance(expression, str) or len(expression) > 256:
        raise ValueError("Expression limited to 256 characters")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("Expression too complex")

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in BINARY:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow):
                if abs(right) > 100 or (left and math.log10(abs(left)) * right > 100):
                    raise ValueError("Exponent too large")
            value = BINARY[type(node.op)](left, right)
        else:
            raise ValueError("Only numeric arithmetic is allowed")
        if type(value) not in {int, float} or not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError("Result outside supported numeric range")
        return value

    return visit(tree.body)


class Tools:
    def __init__(self, home):
        self.home = Path(home).resolve()
        self.files = self.home / "files"
        self.files.mkdir(parents=True, exist_ok=True)
        self.knowledge = Knowledge(self.home / "knowledge.sqlite3")

    def file_path(self, path):
        path = Path(path)
        if path.is_absolute():
            raise ValueError("Use a path relative to workspace/files")
        resolved = (self.files / path).resolve()
        if not resolved.is_relative_to(self.files.resolve()):
            raise ValueError("File path escapes workspace/files")
        return resolved

    def read_file(self, path):
        return read_document(self.file_path(path))

    def write_file(self, path, content):
        if not isinstance(content, str) or len(content.encode("utf-8")) > 1024 * 1024:
            raise ValueError("Content must be text up to 1 MiB")
        target = self.file_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Explicitly refuse overwrites; the caller can choose a new name.
        with target.open("x", encoding="utf-8") as f:
            f.write(content)
        return {"path": str(target), "bytes": target.stat().st_size}

    def list_files(self):
        return [str(p.relative_to(self.files)) for p in sorted(self.files.rglob("*"))
                if p.is_file() and p.resolve().is_relative_to(self.files.resolve())][:1000]

    def registry(self):
        return {
            "calculator": (calculate, {"expression": "arithmetic string"}),
            "get_time": (lambda: datetime.now().astimezone().isoformat(), {}),
            "search_local": (self.knowledge.search, {"query": "text", "limit": "1..20 (optional)"}),
            "search_web": (search_web, {"query": "text", "limit": "1..10 (optional)"}),
            "read_webpage": (read_webpage, {"url": "public HTTP(S) URL"}),
            "save_knowledge": (self.knowledge.add, {"text": "document", "source": "source URL or label",
                                                  "title": "optional title"}),
            "read_file": (self.read_file, {"path": "relative to workspace/files"}),
            "write_file": (self.write_file, {"path": "new relative filename", "content": "text"}),
            "list_files": (self.list_files, {}),
            "run_python": (run_python, {"code": "Python source", "timeout": "1..30 (optional); Docker required"}),
        }

    def describe(self):
        return {name: args for name, (_, args) in self.registry().items()}

    def call(self, name, arguments):
        if name not in self.registry():
            raise ValueError(f"Unknown tool: {name}")
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object")
        function, _ = self.registry()[name]
        return function(**arguments)
