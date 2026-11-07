"""Run a project's own test suite on a throwaway copy (Agent step 6).

The project is copied to a temporary folder (skipping .git, virtualenvs, node_modules,
workspace, caches), the tests run there with a time limit, memory limit, minimal
environment and below-normal priority, and the copy is deleted afterwards. The real
project is never modified by a test run.

This is accident-proofing for the user's own code that the agent edited; it is NOT a
security boundary like a container (network is not blocked). Never point it at code you
do not trust; foreign code goes through runner.run_tests or the Docker sandbox instead.

Detected suites: Python unittest (or pytest when the project uses it), Node's built-in
runner (`node --test`). The summary + first failure are returned for the model to read.
"""
import re
import shutil
import sys
import tempfile
from pathlib import Path

from .projindex import SKIP_DIRS
from .runner import _execute

MAX_COPY_BYTES = 300 * 1024 * 1024
TEST_MEMORY = 1024 * 1024 * 1024
OUTPUT_CAP = 200_000


def _project_size(root):
    total = 0
    for path in Path(root).rglob("*"):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.is_file():
            total += path.stat().st_size
            if total > MAX_COPY_BYTES:
                break
    return total


def detect(root):
    """Pick a test command for the project, or None."""
    root = Path(root)
    py_tests = [p for p in root.rglob("test*.py") if not any(s in p.parts for s in SKIP_DIRS)]
    if py_tests:
        uses_pytest = (root / "pytest.ini").exists() or (root / "conftest.py").exists() or any(
            re.search(r"^\s*import pytest|^\s*from pytest", p.read_text(encoding="utf-8", errors="ignore"), re.M)
            for p in py_tests[:50])
        if uses_pytest:
            return "pytest", [sys.executable, "-m", "pytest", "-q", "-x", "--no-header", "-p", "no:cacheprovider"]
        start = "tests" if (root / "tests").is_dir() else "."
        cmd = [sys.executable, "-m", "unittest", "discover", "-s", start]
        if start == "." or (root / "tests" / "__init__.py").exists():
            cmd += ["-t", "."]  # package-style tests; plain folders rely on PYTHONPATH=<project>
        return "unittest", cmd
    js_tests = [p for p in root.rglob("*.test.js") if not any(s in p.parts for s in SKIP_DIRS)]
    node = shutil.which("node")
    if js_tests and node:
        return "node", [node, "--test"]
    return None


def summarize(kind, output, returncode_hint=None):
    """Parse counts and the first failure from test output."""
    result = {"tests": None, "failed": None}
    if kind == "unittest":
        m = re.search(r"Ran (\d+) tests?", output)
        result["tests"] = int(m.group(1)) if m else None
        m = re.search(r"FAILED \(([^)]*)\)", output)
        if m:
            result["failed"] = sum(int(n) for n in re.findall(r"(?:failures|errors)=(\d+)", m.group(1)))
        elif re.search(r"^OK\b", output, re.M):
            result["failed"] = 0
        first = re.search(r"^(?:FAIL|ERROR): .*?(?=^(?:FAIL|ERROR): |^-{20,}\nRan )", output, re.S | re.M)
    elif kind == "pytest":
        passed = re.search(r"(\d+) passed", output)
        failed = re.search(r"(\d+) (?:failed|error)", output)
        result["failed"] = int(failed.group(1)) if failed else 0
        result["tests"] = (int(passed.group(1)) if passed else 0) + result["failed"]
        first = re.search(r"^_{3,} .*?(?=^_{3,} |^=+ short test summary|\Z)", output, re.S | re.M)
    else:
        m = re.search(r"# tests (\d+)", output)
        result["tests"] = int(m.group(1)) if m else None
        m = re.search(r"# fail (\d+)", output)
        result["failed"] = int(m.group(1)) if m else None
        first = re.search(r"^not ok .*?(?=^(?:not )?ok |^# tests|\Z)", output, re.S | re.M)
    result["first_failure"] = first.group(0).strip()[:3000] if first else ""
    return result


def run_project_tests(root, timeout=120, command=None):
    """Run the project's tests on a temp copy. Returns status/counts/first failure/output tail."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"not a folder: {root}")
    if _project_size(root) > MAX_COPY_BYTES:
        return {"status": "error", "message": "project too large to copy (>300 MB without skipped folders)"}
    found = ("custom", command) if command else detect(root)
    if not found:
        return {"status": "no-tests", "message": "no test files found (test*.py or *.test.js)"}
    kind, cmd = found
    with tempfile.TemporaryDirectory(prefix="lb-proj-") as temp:
        copy = Path(temp) / "project"
        shutil.copytree(root, copy, ignore=shutil.ignore_patterns(*SKIP_DIRS, "*.pyc"))
        env = {"PYTHONPATH": str(copy), "TEMP": temp, "TMP": temp, "HOME": temp, "USERPROFILE": temp,
               "NO_COLOR": "1"}
        output, timed_out = _execute(cmd, copy, timeout, memory=TEST_MEMORY, processes=16,
                                     env_extra=env, output_limit=OUTPUT_CAP)
    output = output.replace("\r\n", "\n").replace(str(copy), "<project>")
    if timed_out:
        return {"status": "timeout", "kind": kind, "message": f"tests did not finish in {timeout}s",
                "output_tail": output[-3000:]}
    info = summarize(kind, output)
    if info["failed"] is None and info["tests"] is None:
        status = "error"
    else:
        status = "pass" if info["failed"] == 0 and (info["tests"] or 0) > 0 else "fail"
    return {"status": status, "kind": kind, **info, "output_tail": output[-3000:]}
