"""Central code library (Agent step 3): ~millions of documented functions, searchable offline.

Built from CodeSearchNet (Hugging Face parquet: code-search-net/code_search_net), so a new
user is not starting from an empty library ("cold start"). Each row keeps its source URL.

LICENSE NOTE: CodeSearchNet does not record per-function licenses (each function carries its
GitHub repo's license). Results are *reference examples* for the agent/model to learn the
approach from, shown with their source link; they are not meant to be pasted verbatim.

Text is read from parquet only; no code from the dataset is ever executed.
Storage: SQLite FTS5 with external content (text stored once), BM25 ranking.
"""
import hashlib
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .projindex import _words, keywords

COLUMNS = ["language", "func_name", "repository_name", "func_path_in_repository",
           "func_code_url", "func_documentation_string", "whole_func_string"]
MIN_CODE, MAX_CODE = 40, 4000
MIN_DOC = 12


def _normalize(code):
    return re.sub(r"\s+", " ", code).strip()


def keep(row):
    """Quality filter: real docstring, reasonable size, not a test/stub."""
    code, doc, name = row["whole_func_string"] or "", row["func_documentation_string"] or "", row["func_name"] or ""
    if not (MIN_CODE <= len(code) <= MAX_CODE) or len(doc.strip()) < MIN_DOC:
        return False
    short = name.rsplit(".", 1)[-1].lower()
    if short.startswith("test") or "/test" in (row["func_path_in_repository"] or "").lower():
        return False
    return True


class CodeLibrary:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS functions (
                    id INTEGER PRIMARY KEY, language TEXT NOT NULL, name TEXT NOT NULL,
                    words TEXT NOT NULL, repo TEXT NOT NULL, path TEXT NOT NULL, url TEXT NOT NULL,
                    doc TEXT NOT NULL, code TEXT NOT NULL, digest TEXT NOT NULL UNIQUE);
                CREATE VIRTUAL TABLE IF NOT EXISTS function_search USING fts5(
                    name, words, doc, code, content='functions', content_rowid='id',
                    tokenize='unicode61');
                CREATE TABLE IF NOT EXISTS sources (name TEXT PRIMARY KEY, rows INTEGER NOT NULL);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def add_rows(self, rows):
        """Insert dict rows (CodeSearchNet column names). Returns (kept, duplicates, filtered)."""
        kept = dup = filtered = 0
        with self.connect() as db:
            for row in rows:
                if not keep(row):
                    filtered += 1
                    continue
                code = row["whole_func_string"].strip()
                digest = hashlib.sha1(_normalize(code).encode("utf-8")).hexdigest()
                name = row["func_name"]
                try:
                    rid = db.execute(
                        "INSERT INTO functions(language,name,words,repo,path,url,doc,code,digest) "
                        "VALUES(?,?,?,?,?,?,?,?,?)",
                        (row["language"], name, _words(name), row["repository_name"],
                         row["func_path_in_repository"], row["func_code_url"],
                         row["func_documentation_string"].strip(), code, digest)).lastrowid
                except sqlite3.IntegrityError:
                    dup += 1
                    continue
                db.execute("INSERT INTO function_search(rowid,name,words,doc,code) VALUES(?,?,?,?,?)",
                           (rid, name, _words(name), row["func_documentation_string"], code))
                kept += 1
        return kept, dup, filtered

    def build_from_parquet(self, paths, languages=None, limit=None, log=print):
        """Stream CodeSearchNet parquet files into the library."""
        import pyarrow.parquet as pq
        total = {"kept": 0, "duplicates": 0, "filtered": 0}
        for path in paths:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=2048, columns=COLUMNS):
                rows = [r for r in batch.to_pylist() if not languages or r["language"] in languages]
                if limit is not None:
                    rows = rows[:max(0, limit - total["kept"])]
                k, d, f = self.add_rows(rows)
                total["kept"] += k
                total["duplicates"] += d
                total["filtered"] += f
                if limit is not None and total["kept"] >= limit:
                    break
            with self.connect() as db:
                db.execute("INSERT OR REPLACE INTO sources(name,rows) VALUES(?,?)",
                           (Path(path).name, pf.metadata.num_rows))
            log(f"{Path(path).name}: total kept {total['kept']:,}")
            if limit is not None and total["kept"] >= limit:
                break
        return total

    def search(self, request, language=None, limit=5):
        if not 1 <= limit <= 50:
            raise ValueError("limit must be 1..50")
        terms = keywords(request)
        if not terms:
            return []
        expression = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
        sql = """SELECT f.language, f.name, f.repo, f.path, f.url, f.doc, f.code,
                        bm25(function_search, 8.0, 6.0, 4.0, 1.0) AS score
                 FROM function_search JOIN functions f ON f.id = function_search.rowid
                 WHERE function_search MATCH ?"""
        params = [expression]
        if language:
            sql += " AND f.language = ?"
            params.append(language)
        sql += " ORDER BY score LIMIT ?"
        params.append(limit)
        with self.connect() as db:
            rows = db.execute(sql, params).fetchall()
        return [dict(r, keywords=terms, license="per source repository (see url)") for r in rows]

    def stats(self):
        with self.connect() as db:
            return {"functions": db.execute("SELECT count(*) FROM functions").fetchone()[0],
                    "by_language": dict(db.execute(
                        "SELECT language, count(*) FROM functions GROUP BY language").fetchall()),
                    "sources": dict(db.execute("SELECT name, rows FROM sources").fetchall())}

    def compact(self):
        with self.connect() as db:
            db.execute("INSERT INTO function_search(function_search) VALUES('optimize')")
        db = sqlite3.connect(self.path)
        try:
            db.execute("VACUUM")
        finally:
            db.close()
