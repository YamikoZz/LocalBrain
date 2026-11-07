import tempfile
import unittest
from pathlib import Path

from localbrain.agent import ScriptedBrain, format_report, run_agent
from localbrain.experience import Experience, error_signature

AUTH = '''import hashlib


def check_password(password, stored):
    return hashlib.sha256(password).hexdigest() == stored


def is_admin(user):
    return user == "admin"
'''
TESTS = '''import hashlib, unittest
from src.auth import check_password, is_admin


class T(unittest.TestCase):
    def test_ok(self):
        self.assertTrue(check_password("pw", hashlib.sha256(b"pw").hexdigest()))

    def test_empty(self):
        self.assertFalse(check_password("", "x"))

    def test_admin(self):
        self.assertTrue(is_admin("admin"))
'''
UNAPPLIABLE = "src/auth.py\n<<<<<<< SEARCH\nthis line is not in the file\n=======\nx\n>>>>>>> REPLACE\n"
WORSE = ('src/auth.py\n<<<<<<< SEARCH\n    return user == "admin"\n=======\n    return False\n'
         '>>>>>>> REPLACE\n')
GOOD = ('src/auth.py\n<<<<<<< SEARCH\n    return hashlib.sha256(password).hexdigest() == stored\n=======\n'
        '    if not password:\n        return False\n'
        '    return hashlib.sha256(password.encode("utf-8")).hexdigest() == stored\n>>>>>>> REPLACE\n')
REQUEST = "login fails with TypeError in check_password"


def make_project(base, with_tests=True):
    root = Path(base)
    (root / "src").mkdir(parents=True)
    (root / "src" / "auth.py").write_text(AUTH, encoding="utf-8")
    if with_tests:
        (root / "tests").mkdir()
        (root / "tests" / "test_auth.py").write_text(TESTS, encoding="utf-8")
    return root


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.kw = dict(index_db=self.home / "idx.sqlite3", experience_db=self.home / "exp.sqlite3",
                       timeout=60, log=lambda *_: None)

    def tearDown(self):
        self.tmp.cleanup()

    def test_error_signature(self):
        sig = error_signature('Traceback...\n  File "<project>/src/a.py", line 5\nTypeError: Strings must be encoded before hashing')
        self.assertEqual(sig, "TypeError: Strings must be encoded before hashing")
        self.assertEqual(error_signature("KeyError: 'name' at 12"), "KeyError: <str> at <n>")
        self.assertEqual(error_signature("all good"), "")

    def test_full_loop_rejects_reverts_then_fixes_and_remembers(self):
        root = make_project(Path(self.tmp.name) / "p1")
        brain = ScriptedBrain([UNAPPLIABLE, WORSE, GOOD])
        report = run_agent(root, REQUEST, brain, **self.kw)
        self.assertEqual(report["status"], "fixed")
        self.assertEqual([r["result"] for r in report["rounds"]],
                         ["patch-rejected", "reverted (tests not better)", "kept"])
        self.assertEqual(report["tests_before"]["failed"], 2)
        self.assertEqual(report["tests_after"]["status"], "pass")
        self.assertEqual(report["changed"], ["src/auth.py"])
        text = (root / "src" / "auth.py").read_text(encoding="utf-8")
        self.assertIn('password.encode("utf-8")', text)
        self.assertIn('return user == "admin"', text)  # the worse patch was reverted
        # steps 1-4: relevant code found and summarized; summary went into the prompt
        self.assertIn("src/auth.py", report["summary"])
        self.assertIn("TypeError", report["summary"])
        self.assertIn("check_password", brain.prompts[0])
        self.assertIn("could not be applied", brain.prompts[1])
        # step 8: remembered
        past = Experience(self.home / "exp.sqlite3").search(REQUEST, report_error := error_signature(
            "TypeError: Strings must be encoded before hashing"))
        self.assertEqual(len(past), 1)
        self.assertIn("SEARCH", past[0]["patch"])
        self.assertIn("Status: fixed", format_report(report))

    def test_second_project_gets_past_fix_in_prompt(self):
        run_agent(make_project(Path(self.tmp.name) / "p1"), REQUEST, ScriptedBrain([GOOD]), **self.kw)
        brain = ScriptedBrain([GOOD])
        report = run_agent(make_project(Path(self.tmp.name) / "p2"), "password check crashes", brain, **self.kw)
        self.assertEqual(report["status"], "fixed")
        self.assertIn("fixed a similar problem before", brain.prompts[0])
        self.assertIn("past fix #1", report["sources"])

    def test_nothing_kept_when_not_fixed(self):
        root = make_project(Path(self.tmp.name) / "p1")
        report = run_agent(root, REQUEST, ScriptedBrain([UNAPPLIABLE, WORSE]), rounds=2, **self.kw)
        self.assertEqual(report["status"], "not-fixed")
        self.assertEqual((root / "src" / "auth.py").read_text(encoding="utf-8"), AUTH)

    def test_no_tests_means_no_change_unless_allowed(self):
        root = make_project(Path(self.tmp.name) / "p1", with_tests=False)
        report = run_agent(root, REQUEST, ScriptedBrain([GOOD]), rounds=1, **self.kw)
        self.assertEqual(report["status"], "not-fixed")
        self.assertEqual((root / "src" / "auth.py").read_text(encoding="utf-8"), AUTH)
        self.assertTrue(any("no tests" in w for w in report["warnings"]))
        report = run_agent(root, REQUEST, ScriptedBrain([GOOD]), rounds=1, allow_unverified=True, **self.kw)
        self.assertEqual(report["status"], "changed-unverified")
        self.assertIn(".encode(", (root / "src" / "auth.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
