"""Experience library (Agent step 8): fixes that made a project's tests pass.

Each entry stores the request, a normalized error signature, the patch that worked and the
files it touched. Next time a similar request or the same error shows up, the agent shows
the old fix to the model (or tries it directly if it still applies). Stays on the user's
machine; nothing is uploaded.
"""
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from .projindex import _words, keywords


def error_signature(text):
    """Last 'SomeError: message' line with paths, numbers and quotes normalized."""
    lines = [l.strip() for l in (text or "").splitlines()
             if re.match(r"^[A-Za-z_][\w.]*(Error|Exception|Warning)\b.*", l.strip())]
    if not lines:
        return ""
    sig = lines[-1]
    sig = re.sub(r"(['\"]).*?\1", "<str>", sig)
    sig = re.sub(r"(<project>)?[\w./\\:-]+\.(py|js|ts)\b", "<file>", sig)
    sig = re.sub(r"\b\d+\b", "<n>", sig)
    return sig[:200]


class Experience:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS fixes (
                    id INTEGER PRIMARY KEY, created REAL NOT NULL, project TEXT NOT NULL,
                    request TEXT NOT NULL, error TEXT NOT NULL, files TEXT NOT NULL,
                    patch TEXT NOT NULL, summary TEXT NOT NULL);
                CREATE VIRTUAL TABLE IF NOT EXISTS fix_search USING fts5(
                    request, error, words, content='fixes', content_rowid='id');
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def add(self, project, request, error, files, patch, summary=""):
        with self.connect() as db:
            rid = db.execute("INSERT INTO fixes(created,project,request,error,files,patch,summary) "
                             "VALUES(?,?,?,?,?,?,?)",
                             (time.time(), str(project), request, error, ",".join(files), patch, summary)).lastrowid
            db.execute("INSERT INTO fix_search(rowid,request,error,words) VALUES(?,?,?,?)",
                       (rid, request, error, _words(request, error, " ".join(files))))
        return rid

    def search(self, request, error="", limit=3):
        terms = keywords(request + " " + error)
        if not terms:
            return []
        expression = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
        with self.connect() as db:
            rows = db.execute("""
                SELECT f.*, bm25(fix_search, 3.0, 6.0, 2.0) AS score FROM fix_search
                JOIN fixes f ON f.id = fix_search.rowid WHERE fix_search MATCH ?
                ORDER BY score LIMIT ?""", (expression, limit)).fetchall()
        out = [dict(r) for r in rows]
        if error:  # exact error signature first
            out.sort(key=lambda r: (r["error"] != error, r["score"]))
        return out

    def count(self):
        with self.connect() as db:
            return db.execute("SELECT count(*) FROM fixes").fetchone()[0]
