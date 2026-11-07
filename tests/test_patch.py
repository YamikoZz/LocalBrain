import tempfile
import textwrap
import unittest
from pathlib import Path

from localbrain.patch import PatchError, apply_patch, parse, revert
from localbrain.projtest import detect, run_project_tests, summarize

AUTH = '''import hashlib


def check_password(password, stored):
    return hashlib.sha256(password).hexdigest() == stored
'''

FIX = '''src/auth.py
<<<<<<< SEARCH
    return hashlib.sha256(password).hexdigest() == stored
=======
    if not password:
        return False
    return hashlib.sha256(password.encode("utf-8")).hexdigest() == stored
>>>>>>> REPLACE
'''


class PatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "src").mkdir()
        self.auth = self.root / "src" / "auth.py"
        self.auth.write_text(AUTH, encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_parse_with_fences_and_multiple_blocks(self):
        text = "Here is the fix:\n```python\n" + FIX + "```\n" + FIX.replace("src/auth.py", "b.py")
        blocks = parse(text)
        self.assertEqual([b["path"] for b in blocks], ["src/auth.py", "b.py"])
        with self.assertRaises(PatchError):
            parse("no blocks here")

    def test_exact_apply_diff_and_revert(self):
        result = apply_patch(self.root, FIX)
        self.assertEqual(result["files"], ["src/auth.py"])
        self.assertIn('+    return hashlib.sha256(password.encode("utf-8"))', result["diff"])
        self.assertIn("if not password", self.auth.read_text(encoding="utf-8"))
        revert(result["backup"])
        self.assertEqual(self.auth.read_text(encoding="utf-8"), AUTH)

    def test_dry_run_does_not_write(self):
        result = apply_patch(self.root, FIX, dry_run=True)
        self.assertTrue(result["diff"])
        self.assertEqual(self.auth.read_text(encoding="utf-8"), AUTH)

    def test_indentation_tolerant_match_is_reindented(self):
        loose = FIX.replace("    return hashlib.sha256(password).hexdigest() == stored\n=======",
                            "return hashlib.sha256(password).hexdigest() == stored\n=======")
        loose = loose.replace("    if not password:\n        return False\n    return hashlib.sha256(password.encode",
                              "if not password:\n    return False\nreturn hashlib.sha256(password.encode")
        apply_patch(self.root, loose)
        text = self.auth.read_text(encoding="utf-8")
        self.assertIn("\n    if not password:\n        return False\n    return hashlib", text)

    def test_create_file_and_all_or_nothing(self):
        good_new = "src/util.py\n<<<<<<< SEARCH\n=======\nX = 1\n>>>>>>> REPLACE\n"
        bad = "src/auth.py\n<<<<<<< SEARCH\nnot in file\n=======\nY\n>>>>>>> REPLACE\n"
        with self.assertRaises(PatchError) as ctx:
            apply_patch(self.root, good_new + bad)
        self.assertIn("block 2", str(ctx.exception))
        self.assertFalse((self.root / "src" / "util.py").exists())  # nothing written
        result = apply_patch(self.root, good_new)
        self.assertEqual((self.root / "src" / "util.py").read_text(encoding="utf-8"), "X = 1\n")
        revert(result["backup"])
        self.assertFalse((self.root / "src" / "util.py").exists())

    def test_ambiguous_and_unsafe_paths_rejected(self):
        self.auth.write_text("x = 1\nx = 1\n", encoding="utf-8")
        with self.assertRaises(PatchError):
            apply_patch(self.root, "src/auth.py\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE\n")
        for bad in ["../evil.py", "/etc/passwd", "C:/x.py", ".git/config"]:
            with self.assertRaises(PatchError):
                apply_patch(self.root, f"{bad}\n<<<<<<< SEARCH\n=======\nx\n>>>>>>> REPLACE\n")


TEST_OK = textwrap.dedent('''
    import unittest
    from calc import add

    class T(unittest.TestCase):
        def test_add(self):
            self.assertEqual(add(2, 3), 5)

        def test_neg(self):
            self.assertEqual(add(-1, 1), 0)
''')


class ProjectTestRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "tests").mkdir()
        (self.root / "tests" / "test_calc.py").write_text(TEST_OK, encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_detect(self):
        self.assertEqual(detect(self.root)[0], "unittest")
        (self.root / "conftest.py").write_text("", encoding="utf-8")
        self.assertEqual(detect(self.root)[0], "pytest")

    def test_pass_then_fail_with_first_failure(self):
        (self.root / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        ok = run_project_tests(self.root, timeout=60)
        self.assertEqual((ok["status"], ok["tests"], ok["failed"]), ("pass", 2, 0))
        (self.root / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        bad = run_project_tests(self.root, timeout=60)
        self.assertEqual((bad["status"], bad["failed"]), ("fail", 2))
        self.assertIn("AssertionError", bad["first_failure"])
        self.assertNotIn(self.tmp.name, bad["output_tail"])  # temp path hidden
        self.assertEqual((self.root / "calc.py").read_text(encoding="utf-8").count("a - b"), 1)

    def test_timeout_and_no_tests(self):
        (self.root / "calc.py").write_text("def add(a, b):\n    while True:\n        pass\n", encoding="utf-8")
        self.assertEqual(run_project_tests(self.root, timeout=3)["status"], "timeout")
        with tempfile.TemporaryDirectory() as empty:
            self.assertEqual(run_project_tests(empty)["status"], "no-tests")

    def test_summarize_pytest_and_node(self):
        py = summarize("pytest", "____ test_x ____\nassert 1 == 2\n=== short test summary\n1 failed, 3 passed in 0.1s")
        self.assertEqual((py["tests"], py["failed"]), (4, 1))
        self.assertIn("assert 1 == 2", py["first_failure"])
        node = summarize("node", "ok 1 - a\nnot ok 2 - b\n  expected 1\n# tests 2\n# fail 1\n")
        self.assertEqual((node["tests"], node["failed"]), (2, 1))
        self.assertTrue(node["first_failure"].startswith("not ok 2"))


if __name__ == "__main__":
    unittest.main()
