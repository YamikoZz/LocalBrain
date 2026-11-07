"""Project index (Agent step 1-2): split a code project into functions/classes and search them.

The agent uses this before writing code: pull keywords out of the user's request
("login fails with TypeError in check_password") and find the functions that matter,
so the model reads a few relevant snippets instead of whole files.

* SQLite FTS5 (BM25), one file, no server. Identifiers are also stored split into words
  (check_password -> "check password", getUserName -> "get user name") so plain-English
  requests match code names.
* Incremental: only files whose size/mtime changed are re-parsed; deleted files are dropped.
* Python is split with `ast` (parse only, nothing is executed). Other languages use
  declaration regexes; a symbol runs until the next symbol (capped), which is enough to
  locate code. Files with no symbols are indexed as one "module" entry.
"""
import ast
import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path

LANGUAGES = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".cjs": "javascript", ".ts": "typescript", ".tsx": "typescript", ".lua": "lua",
    ".luau": "lua", ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
    ".cs": "csharp", ".java": "java", ".go": "go", ".rs": "rust", ".rb": "ruby",
    ".php": "php", ".sh": "shell", ".sql": "sql", ".html": "html", ".css": "css",
}
SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env", "__pycache__",
             "dist", "build", ".mypy_cache", ".pytest_cache", ".idea", ".vscode", "workspace",
             ".localbrain", "target", "vendor", ".next", "coverage"}
MAX_FILE_BYTES = 1_000_000
MAX_SYMBOL_LINES = 200
MODULE_LINES = 60

_NAME = r"([A-Za-z_$][\w$]*)"
PATTERNS = {
    "javascript": [
        ("function", re.compile(rf"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*{_NAME}\s*\(")),
        ("class", re.compile(rf"^\s*(?:export\s+)?(?:default\s+)?class\s+{_NAME}")),
        ("function", re.compile(rf"^\s*(?:export\s+)?(?:const|let|var)\s+{_NAME}\s*=\s*(?:async\s+)?(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)")),
        ("method", re.compile(rf"^\s+(?:static\s+)?(?:async\s+)?{_NAME}\s*\([^)]*\)\s*\{{")),
    ],
    "lua": [
        ("function", re.compile(r"^\s*(?:local\s+)?function\s+([\w.:]+)\s*\(")),
        ("function", re.compile(r"^\s*(?:local\s+)?([\w.]+)\s*=\s*function\s*\(")),
    ],
    "c": [("function", re.compile(rf"^[A-Za-z_][\w\s\*&<>:,]*?\b{_NAME}\s*\([^;]*\)\s*(?:const\s*)?\{{?\s*$"))],
    "go": [("function", re.compile(r"^func\s+(?:\([^)]*\)\s*)?(\w+)\s*\(")),
           ("type", re.compile(r"^type\s+(\w+)\s+(?:struct|interface)"))],
    "rust": [("function", re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\w+)")),
             ("type", re.compile(r"^\s*(?:pub\s+)?(?:struct|enum|trait|impl)\s+(\w+)"))],
    "ruby": [("function", re.compile(r"^\s*def\s+(?:self\.)?([\w?!]+)")),
             ("class", re.compile(r"^\s*(?:class|module)\s+([\w:]+)"))],
    "php": [("function", re.compile(r"^\s*(?:public|private|protected|static|\s)*function\s+(\w+)\s*\(")),
            ("class", re.compile(r"^\s*(?:abstract\s+|final\s+)?class\s+(\w+)"))],
    "shell": [("function", re.compile(r"^\s*(?:function\s+)?([\w-]+)\s*\(\)\s*\{?"))],
    "sql": [("table", re.compile(r"(?i)^\s*create\s+(?:table|view|function|procedure)\s+(?:if\s+not\s+exists\s+)?([\w.\"`]+)"))],
}
PATTERNS["typescript"] = PATTERNS["javascript"] + [
    ("type", re.compile(rf"^\s*(?:export\s+)?(?:interface|type|enum)\s+{_NAME}"))]
PATTERNS["cpp"] = PATTERNS["c"] + [("class", re.compile(r"^\s*(?:class|struct)\s+(\w+)"))]
PATTERNS["csharp"] = PATTERNS["c"] + [("class", re.compile(r"^\s*(?:public\s+|internal\s+)?(?:static\s+)?(?:class|struct|interface)\s+(\w+)"))]
PATTERNS["java"] = PATTERNS["csharp"]
C_KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "catch", "else", "do", "new", "delete"}

STOPWORDS = set("""a an the and or but if then else for to of in on at by with from into onto is are was were be been
being it its this that these those there here i me my we our you your he she they them his her their what which who
whom how why when where please can could would should will shall may might must do does did done make makes made
write create add fix change update help need needs want wants get got give use using used not no yes so as than
too very just only also some any all each every one two function functions method code file files line lines error
errors bug bugs work works working broken issue problem""".split())


def split_identifier(name):
    """check_password -> [check, password]; getUserName -> [get, user, name]; HTTPServer -> [http, server]."""
    parts = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|\b|_)|[A-Z]?[a-z]+|[A-Z]+|\d+", name)
    return [p.lower() for p in parts if p]


def keywords(request, limit=16):
    """Keywords for code search from a natural-language request (identifiers, error names, words)."""
    found = []
    found += re.findall(r"`([^`]{1,80})`", request)
    found += re.findall(r"\b[\w./-]+\.(?:py|js|ts|tsx|lua|luau|cpp|c|h|go|rs|rb|php|java|sql)\b", request)
    found += re.findall(r"\b[A-Za-z_][\w]*\b", request)
    out = []
    for item in found:
        stem = Path(item).stem if "." in item else item
        for token in [stem, *split_identifier(stem)]:
            token = token.strip().lower()
            if len(token) >= 3 and token not in STOPWORDS and not token.isdigit() and token not in out:
                out.append(token)
    return out[:limit]


def _words(*texts):
    words = []
    for text in texts:
        for ident in re.findall(r"[A-Za-z_][\w]*", text or ""):
            words.append(ident.lower())
            words += split_identifier(ident)
    return " ".join(dict.fromkeys(words))


def _python_symbols(text):
    tree = ast.parse(text)  # parse only; the file is never executed
    out = []

    def visit(nodes, prefix):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kind = "class" if isinstance(node, ast.ClassDef) else ("method" if prefix else "function")
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                out.append({"name": prefix + node.name, "kind": kind, "start": start,
                            "end": node.end_lineno or node.lineno, "doc": ast.get_docstring(node) or ""})
                if isinstance(node, ast.ClassDef):
                    visit(node.body, prefix + node.name + ".")
    visit(tree.body, "")
    return out


def _regex_symbols(text, language):
    lines = text.splitlines()
    found = []
    for i, line in enumerate(lines):
        for kind, pattern in PATTERNS.get(language, []):
            m = pattern.match(line)
            if m and m.group(1) not in C_KEYWORDS:
                found.append({"name": m.group(1).strip('"`'), "kind": kind, "start": i + 1})
                break
    for j, sym in enumerate(found):
        nxt = found[j + 1]["start"] - 1 if j + 1 < len(found) else len(lines)
        sym["end"] = max(sym["start"], min(nxt, sym["start"] + MAX_SYMBOL_LINES - 1))
        sym["doc"] = ""
    return found


def extract_symbols(text, language):
    if language == "python":
        try:
            return _python_symbols(text)
        except (SyntaxError, ValueError):
            return []
    return _regex_symbols(text, language)


class ProjectIndex:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE, language TEXT NOT NULL,
                    size INTEGER NOT NULL, mtime REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS symbols (
                    id INTEGER PRIMARY KEY,
                    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                    name TEXT NOT NULL, kind TEXT NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL,
                    signature TEXT NOT NULL, body TEXT NOT NULL);
                CREATE VIRTUAL TABLE IF NOT EXISTS symbol_search USING
                    fts5(name, words, signature, doc, path, body, tokenize='unicode61');
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def _drop_file(self, db, file_id):
        db.execute("DELETE FROM symbol_search WHERE rowid IN (SELECT id FROM symbols WHERE file_id=?)",
                   (file_id,))
        db.execute("DELETE FROM files WHERE id=?", (file_id,))

    def index(self, root):
        """(Re)index source files under root. Returns counts of scanned/updated/removed files."""
        root = Path(root).resolve()
        seen, updated, symbols = set(), 0, 0
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('root',?)", (str(root),))
            known = {r["path"]: r for r in db.execute("SELECT id,path,size,mtime FROM files")}
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
                for name in sorted(filenames):
                    language = LANGUAGES.get(Path(name).suffix.lower())
                    if not language:
                        continue
                    full = Path(dirpath) / name
                    try:
                        st = full.stat()
                    except OSError:
                        continue
                    if st.st_size > MAX_FILE_BYTES:
                        continue
                    rel = full.relative_to(root).as_posix()
                    seen.add(rel)
                    old = known.get(rel)
                    if old and old["size"] == st.st_size and old["mtime"] == st.st_mtime:
                        continue
                    try:
                        text = full.read_text(encoding="utf-8")
                    except (UnicodeDecodeError, OSError):
                        continue
                    if old:
                        self._drop_file(db, old["id"])
                    symbols += self._add_file(db, rel, language, st, text)
                    updated += 1
            removed = [r for p, r in known.items() if p not in seen]
            for r in removed:
                self._drop_file(db, r["id"])
        return {"root": str(root), "files": len(seen), "updated": updated,
                "removed": len(removed), "symbols_added": symbols}

    def _add_file(self, db, rel, language, st, text):
        file_id = db.execute("INSERT INTO files(path,language,size,mtime) VALUES(?,?,?,?)",
                             (rel, language, st.st_size, st.st_mtime)).lastrowid
        lines = text.splitlines()
        found = extract_symbols(text, language) or [
            {"name": Path(rel).stem, "kind": "module", "start": 1,
             "end": min(len(lines), MODULE_LINES) or 1, "doc": ""}]
        for sym in found:
            body = "\n".join(lines[sym["start"] - 1:sym["end"]])
            signature = next((l.strip() for l in lines[sym["start"] - 1:sym["end"]]
                              if l.strip() and not l.strip().startswith("@")), sym["name"])[:200]
            sid = db.execute(
                "INSERT INTO symbols(file_id,name,kind,start,end,signature,body) VALUES(?,?,?,?,?,?,?)",
                (file_id, sym["name"], sym["kind"], sym["start"], sym["end"], signature, body)).lastrowid
            db.execute("INSERT INTO symbol_search(rowid,name,words,signature,doc,path,body) VALUES(?,?,?,?,?,?,?)",
                       (sid, sym["name"], _words(sym["name"], rel), signature, sym["doc"], rel, body))
        return len(found)

    def search(self, request, limit=8):
        """Find symbols for a natural-language request or identifier. Best match first."""
        if not 1 <= limit <= 50:
            raise ValueError("limit must be 1..50")
        terms = keywords(request)
        if not terms:
            return []
        expression = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
        with self.connect() as db:
            rows = db.execute("""
                SELECT s.name, s.kind, f.path, f.language, s.start, s.end, s.signature, s.body,
                       bm25(symbol_search, 10.0, 6.0, 3.0, 2.0, 2.0, 1.0) AS score
                FROM symbol_search JOIN symbols s ON s.id = symbol_search.rowid
                JOIN files f ON f.id = s.file_id
                WHERE symbol_search MATCH ? ORDER BY score LIMIT ?
            """, (expression, limit)).fetchall()
        return [dict(r, keywords=terms) for r in rows]

    def stats(self):
        with self.connect() as db:
            root = db.execute("SELECT value FROM meta WHERE key='root'").fetchone()
            return {"root": root[0] if root else None,
                    "files": db.execute("SELECT count(*) FROM files").fetchone()[0],
                    "symbols": db.execute("SELECT count(*) FROM symbols").fetchone()[0],
                    "by_language": dict(db.execute(
                        "SELECT language, count(*) FROM files GROUP BY language ORDER BY 2 DESC").fetchall())}

    def compact(self):
        """Shrink the database file after many re-indexes."""
        with self.connect() as db:
            db.execute("INSERT INTO symbol_search(symbol_search) VALUES('optimize')")
        db = sqlite3.connect(self.path)
        try:
            db.execute("VACUUM")
        finally:
            db.close()
