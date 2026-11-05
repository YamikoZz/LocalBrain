import hashlib
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class Knowledge:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY, source TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL, sha256 TEXT NOT NULL,
                    fetched_at TEXT NOT NULL, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL, body TEXT NOT NULL,
                    UNIQUE(document_id, ordinal));
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_search USING
                    fts5(body, title, tokenize='trigram');
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

    def add(self, text, source, title=""):
        text = text.strip()
        if not text or not source.strip():
            raise ValueError("Nonempty text and source required")
        if len(text) > 4_000_000:
            raise ValueError("Document exceeds 4 million characters")
        title = title.strip() or source
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            existing = db.execute(
                "SELECT id, sha256 FROM documents WHERE source=?", (source,)).fetchone()
            if existing and existing["sha256"] == digest:
                db.execute("UPDATE documents SET fetched_at=? WHERE id=?", (now, existing["id"]))
                return {"id": existing["id"], "changed": False}
            if existing:
                doc_id = existing["id"]
                db.execute("DELETE FROM chunk_search WHERE rowid IN "
                           "(SELECT id FROM chunks WHERE document_id=?)", (doc_id,))
                db.execute("DELETE FROM chunks WHERE document_id=?", (doc_id,))
                db.execute("UPDATE documents SET title=?,sha256=?,fetched_at=?,body=? WHERE id=?",
                           (title, digest, now, text, doc_id))
            else:
                doc_id = db.execute(
                    "INSERT INTO documents(source,title,sha256,fetched_at,body) VALUES(?,?,?,?,?)",
                    (source, title, digest, now, text)).lastrowid
            for ordinal, start in enumerate(range(0, len(text), 580)):
                body = text[start:start + 700]
                chunk_id = db.execute(
                    "INSERT INTO chunks(document_id,ordinal,body) VALUES(?,?,?)",
                    (doc_id, ordinal, body)).lastrowid
                db.execute("INSERT INTO chunk_search(rowid,body,title) VALUES(?,?,?)",
                           (chunk_id, body, title))
        return {"id": doc_id, "changed": True}

    def search(self, query, limit=5):
        if not 1 <= limit <= 20:
            raise ValueError("limit must be 1..20")
        terms = list(dict.fromkeys(
            word for word in re.findall(r"[^\W_]+", query, flags=re.UNICODE) if len(word) >= 3))[:16]
        if not terms:
            return []
        expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        with self.connect() as db:
            rows = db.execute("""
                SELECT c.id AS chunk_id,c.body,d.source,d.title,d.fetched_at,
                       bm25(chunk_search) AS score
                FROM chunk_search JOIN chunks c ON c.id=chunk_search.rowid
                JOIN documents d ON d.id=c.document_id
                WHERE chunk_search MATCH ? ORDER BY score LIMIT ?
            """, (expression, limit)).fetchall()
            if not rows:
                # Thai has no mandatory spaces: fall back to bounded character trigrams.
                grams = list(dict.fromkeys(
                    term[i:i + 3] for term in terms
                    for i in range(0, min(len(term) - 2, 48), 3)))[:24]
                fallback = " OR ".join('"' + word.replace('"', '""') + '"' for word in grams)
                if fallback:
                    rows = db.execute("""
                        SELECT c.id AS chunk_id,c.body,d.source,d.title,d.fetched_at,
                               bm25(chunk_search) AS score
                        FROM chunk_search JOIN chunks c ON c.id=chunk_search.rowid
                        JOIN documents d ON d.id=c.document_id
                        WHERE chunk_search MATCH ? ORDER BY score LIMIT ?
                    """, (fallback, limit)).fetchall()
        return [dict(row) for row in rows]

    def sources(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute(
                "SELECT source,title,fetched_at FROM documents ORDER BY id")]

    def stats(self):
        with self.connect() as db:
            return {"documents": db.execute("SELECT count(*) FROM documents").fetchone()[0],
                    "chunks": db.execute("SELECT count(*) FROM chunks").fetchone()[0]}
