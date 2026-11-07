import os
import tempfile
import time
import unittest
from pathlib import Path

from localbrain.projindex import ProjectIndex, extract_symbols, keywords, split_identifier

PY = '''import hashlib

def check_password(password, stored):
    """Compare a password with its stored hash."""
    return hashlib.sha256(password).hexdigest() == stored


class UserStore:
    def get_user(self, name):
        return self.users[name]

    @staticmethod
    def save_user(user):
        pass
'''

JS = '''export function renderLoginForm(el) {
  el.innerHTML = "<form></form>";
}

const fetchProfile = async (id) => {
  return fetch("/api/profile/" + id);
};

class Cart {
  addItem(item) {
    this.items.push(item);
  }
}
'''

LUA = '''local function add(a, b)
  return a + b
end

function Inventory.removeItem(slot)
  Inventory.slots[slot] = nil
end
'''


class SplitAndKeywords(unittest.TestCase):
    def test_split_identifier(self):
        self.assertEqual(split_identifier("check_password"), ["check", "password"])
        self.assertEqual(split_identifier("getUserName"), ["get", "user", "name"])
        self.assertEqual(split_identifier("HTTPServer"), ["http", "server"])

    def test_keywords_drop_filler_keep_identifiers(self):
        kw = keywords("Please fix the login, it fails with TypeError in `check_password` (auth.py)")
        for want in ["login", "typeerror", "check_password", "check", "password", "auth"]:
            self.assertIn(want, kw)
        for filler in ["please", "fix", "the", "with"]:
            self.assertNotIn(filler, kw)


class Extraction(unittest.TestCase):
    def test_python_functions_classes_methods(self):
        syms = {s["name"]: s for s in extract_symbols(PY, "python")}
        self.assertEqual(set(syms), {"check_password", "UserStore", "UserStore.get_user", "UserStore.save_user"})
        self.assertEqual(syms["check_password"]["kind"], "function")
        self.assertEqual(syms["UserStore.get_user"]["kind"], "method")
        self.assertIn("stored hash", syms["check_password"]["doc"])
        self.assertEqual(syms["UserStore.save_user"]["start"], 12)  # starts at the decorator

    def test_python_syntax_error_is_not_fatal(self):
        self.assertEqual(extract_symbols("def broken(:\n", "python"), [])

    def test_javascript_and_lua(self):
        js = {s["name"] for s in extract_symbols(JS, "javascript")}
        self.assertTrue({"renderLoginForm", "fetchProfile", "Cart", "addItem"} <= js)
        lua = {s["name"] for s in extract_symbols(LUA, "lua")}
        self.assertEqual(lua, {"add", "Inventory.removeItem"})


class IndexAndSearch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "proj"
        (self.root / "src").mkdir(parents=True)
        (self.root / "node_modules" / "lib").mkdir(parents=True)
        (self.root / "src" / "auth.py").write_text(PY, encoding="utf-8")
        (self.root / "src" / "ui.js").write_text(JS, encoding="utf-8")
        (self.root / "game.lua").write_text(LUA, encoding="utf-8")
        (self.root / "notes.txt").write_text("not code", encoding="utf-8")
        (self.root / "node_modules" / "lib" / "x.js").write_text("function vendored() {}", encoding="utf-8")
        self.index = ProjectIndex(Path(self.tmp.name) / "idx.sqlite3")

    def tearDown(self):
        self.tmp.cleanup()

    def test_index_skips_vendor_and_non_code(self):
        r = self.index.index(self.root)
        self.assertEqual((r["files"], r["updated"]), (3, 3))
        self.assertEqual(self.index.stats()["by_language"], {"javascript": 1, "lua": 1, "python": 1})

    def test_search_finds_relevant_function_first(self):
        self.index.index(self.root)
        hits = self.index.search("login fails with TypeError when checking the password")
        self.assertEqual(hits[0]["name"], "check_password")
        self.assertEqual(hits[0]["path"], "src/auth.py")
        self.assertIn("sha256", hits[0]["body"])
        self.assertEqual(self.index.search("render the login form")[0]["name"], "renderLoginForm")
        self.assertEqual(self.index.search("remove item from inventory")[0]["name"], "Inventory.removeItem")
        self.assertEqual(self.index.search("zzz qqq"), [])

    def test_incremental_update_and_delete(self):
        self.index.index(self.root)
        again = self.index.index(self.root)
        self.assertEqual(again["updated"], 0)
        target = self.root / "game.lua"
        target.write_text(LUA + "\nfunction spawnEnemy(x)\n  return x\nend\n", encoding="utf-8")
        later = time.time() + 5
        os.utime(target, (later, later))
        (self.root / "src" / "ui.js").unlink()
        r = self.index.index(self.root)
        self.assertEqual((r["updated"], r["removed"]), (1, 1))
        self.assertEqual(self.index.search("spawn enemy")[0]["name"], "spawnEnemy")
        self.assertEqual(self.index.search("renderLoginForm"), [])
        self.index.compact()


if __name__ == "__main__":
    unittest.main()
