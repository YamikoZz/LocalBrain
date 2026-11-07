import json
import tempfile
import unittest
from pathlib import Path

from localbrain.codelib import CodeLibrary, keep
from localbrain.stackexchange import StackExchange, html_to_text


def row(name, doc, code, lang="python", path="pkg/mod.py"):
    return {"language": lang, "func_name": name, "repository_name": "user/repo",
            "func_path_in_repository": path, "func_code_url": f"https://github.com/user/repo/blob/x/{path}#L1",
            "func_documentation_string": doc, "whole_func_string": code}


HASH = row("hash_password", "Hash a password with bcrypt and a random salt.",
           "def hash_password(password):\n    salt = bcrypt.gensalt()\n    return bcrypt.hashpw(password.encode(), salt)\n")
REVERSE = row("reverseString", "Reverse the characters of a string.",
              "function reverseString(s) {\n  return s.split('').reverse().join('');\n}", lang="javascript",
              path="src/str.js")
READ_JSON = row("load_config", "Read a JSON configuration file and return a dict.",
                "def load_config(path):\n    with open(path) as fh:\n        return json.load(fh)\n")


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lib = CodeLibrary(Path(self.tmp.name) / "lib.sqlite3")

    def tearDown(self):
        self.tmp.cleanup()

    def test_filter(self):
        self.assertTrue(keep(HASH))
        self.assertFalse(keep(row("f", "short", "def f():\n    return 1234567890123456789012345678901234\n")))
        self.assertFalse(keep(row("test_hash", "Check hashing works properly.", HASH["whole_func_string"])))
        self.assertFalse(keep(row("x", "A valid long docstring here.", "x" * 5000)))

    def test_add_dedup_and_search(self):
        kept, dup, filtered = self.lib.add_rows([HASH, REVERSE, READ_JSON, dict(HASH, func_name="dup")])
        self.assertEqual((kept, dup, filtered), (3, 1, 0))
        top = self.lib.search("how do I hash a password securely")[0]
        self.assertEqual(top["name"], "hash_password")
        self.assertIn("github.com", top["url"])
        self.assertEqual(self.lib.search("reverse a string", language="javascript")[0]["name"], "reverseString")
        self.assertEqual(self.lib.search("reverse a string", language="python"), [])
        self.assertEqual(self.lib.search("read json config file")[0]["name"], "load_config")
        self.assertEqual(self.lib.stats()["functions"], 3)
        self.lib.compact()

    def test_build_from_parquet(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = Path(self.tmp.name) / "train-0.parquet"
        pq.write_table(pa.Table.from_pylist([HASH, REVERSE, READ_JSON]), path)
        result = self.lib.build_from_parquet([path], languages=["python"], log=lambda *_: None)
        self.assertEqual(result["kept"], 2)
        self.assertEqual(self.lib.stats()["sources"], {"train-0.parquet": 3})


ANSWER_HTML = ("<p>Encode the string first:</p><pre><code>hashlib.sha256(s.encode(&quot;utf-8&quot;))"
               "</code></pre><p>Strings are unicode in Python 3.</p>")


class FakeAPI:
    def __init__(self):
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        if "/search/advanced" in url:
            return {"items": [{"title": "TypeError: Unicode-objects must be encoded before hashing",
                               "link": "https://stackoverflow.com/q/1", "tags": ["python", "hashlib"],
                               "accepted_answer_id": 11},
                              {"title": "no accepted answer", "link": "https://stackoverflow.com/q/2"}],
                    "quota_remaining": 299}
        if "/answers/11" in url:
            return {"items": [{"answer_id": 11, "score": 42, "body": ANSWER_HTML,
                               "owner": {"display_name": "alice"}}], "quota_remaining": 298, "backoff": 0}
        raise AssertionError(url)


class StackExchangeTests(unittest.TestCase):
    def test_html_to_text(self):
        text, codes = html_to_text(ANSWER_HTML)
        self.assertEqual(codes, ['hashlib.sha256(s.encode("utf-8"))'])
        self.assertIn("Encode the string first", text)
        self.assertNotIn("<p>", text)
        self.assertIn('```\nhashlib.sha256(s.encode("utf-8"))\n```', text)

    def test_search_cache_and_key_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            api = FakeAPI()
            se = StackExchange(Path(tmp) / "c.sqlite3", fetch=api, key="SECRET")
            res = se.search("Unicode-objects must be encoded before hashing", tagged="python")
            self.assertEqual(len(res), 1)
            self.assertEqual(res[0]["author"], "alice")
            self.assertEqual(res[0]["license"], "CC BY-SA 4.0")
            self.assertIn("encode", res[0]["code"][0])
            self.assertEqual(se.quota_remaining, 298)
            self.assertTrue(all("key=SECRET" in u for u in api.calls))
            se.search("Unicode-objects must be encoded before hashing", tagged="python")
            self.assertEqual(len(api.calls), 2)  # second search served from cache
            import sqlite3
            db = sqlite3.connect(Path(tmp) / "c.sqlite3")
            try:
                stored = db.execute("SELECT url, body FROM cache").fetchall()
            finally:
                db.close()
            self.assertFalse(any("SECRET" in u or "SECRET" in b for u, b in stored))

    def test_api_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            se = StackExchange(Path(tmp) / "c.sqlite3", fetch=lambda url: {"error_id": 502, "error_message": "throttle"})
            with self.assertRaises(RuntimeError):
                se.search("anything")


if __name__ == "__main__":
    unittest.main()
