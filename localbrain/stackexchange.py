"""Live Q&A search via the public Stack Exchange API (Agent step 3, online fallback).

Used when the project index and the offline library don't cover a request, e.g. a new error
message. Returns accepted answers as text plus their code blocks, with link and author for
attribution (content is CC BY-SA 4.0).

* No API key needed (~300 requests/day per IP); a key, if given, raises that to 10,000.
  Never store keys in the repo: pass via the STACKEXCHANGE_KEY environment variable.
* Responses are cached in SQLite (default 30 days) so repeated questions cost nothing.
* Honors the API's `backoff` field; answers are text only, nothing is executed.
"""
import gzip
import html
import json
import os
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path

API = "https://api.stackexchange.com/2.3"
USER_AGENT = "LocalBrainResearch/0.1"
LICENSE = "CC BY-SA 4.0"


def http_get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(request, timeout=20) as response:
        raw = response.read(4 * 1024 * 1024)
        if response.headers.get("Content-Encoding") == "gzip" or raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def html_to_text(body):
    """Answer HTML -> (plain text, [code blocks])."""
    codes = []

    def fence(match):
        code = html.unescape(re.sub(r"<[^>]+>", "", match.group(1))).strip("\n")
        codes.append(code)
        return f"\n\x00{len(codes) - 1}\x00\n"

    text = re.sub(r"<pre[^>]*>\s*<code[^>]*>(.*?)</code>\s*</pre>", fence, body or "", flags=re.S)
    text = re.sub(r"<br\s*/?>|</p>|</li>|</h\d>", "\n", text)
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    text = re.sub(r"\x00(\d+)\x00", lambda m: "```\n" + codes[int(m.group(1))] + "\n```", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip(), codes


class StackExchange:
    def __init__(self, cache_path, fetch=http_get_json, key=None, ttl_days=30, site="stackoverflow"):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.fetch = fetch
        self.key = key if key is not None else os.environ.get("STACKEXCHANGE_KEY")
        self.ttl = ttl_days * 86400
        self.site = site
        self.quota_remaining = None
        self._not_before = 0.0
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS cache (url TEXT PRIMARY KEY, at REAL NOT NULL, body TEXT NOT NULL)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.cache_path, timeout=15)
        try:
            with db:
                yield db
        finally:
            db.close()

    def _get(self, path, **params):
        params = {"site": self.site, **params}
        if self.key:
            params["key"] = self.key
        url = f"{API}/{path}?{urllib.parse.urlencode(sorted(params.items()))}"
        cache_key = re.sub(r"([?&])key=[^&]*&?", r"\1", url)  # never persist the key
        with self.connect() as db:
            row = db.execute("SELECT at, body FROM cache WHERE url=?", (cache_key,)).fetchone()
        if row and time.time() - row[0] < self.ttl:
            return json.loads(row[1])
        wait = self._not_before - time.time()
        if wait > 0:
            time.sleep(min(wait, 30))
        data = self.fetch(url)
        if "error_id" in data:
            raise RuntimeError(f"Stack Exchange API error {data['error_id']}: {data.get('error_message', '')}")
        self.quota_remaining = data.get("quota_remaining", self.quota_remaining)
        if data.get("backoff"):
            self._not_before = time.time() + int(data["backoff"])
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO cache(url, at, body) VALUES(?,?,?)",
                       (cache_key, time.time(), json.dumps(data)))
        return data

    def search(self, query, tagged=None, limit=3):
        """Questions matching `query` that have an accepted answer, best first, with that answer."""
        if not query.strip():
            return []
        params = {"order": "desc", "sort": "relevance", "q": query[:200], "accepted": "True",
                  "pagesize": max(1, min(limit, 10)), "filter": "withbody"}
        if tagged:
            params["tagged"] = tagged
        questions = self._get("search/advanced", **params).get("items", [])
        ids = [str(q["accepted_answer_id"]) for q in questions if q.get("accepted_answer_id")]
        answers = {}
        if ids:
            for a in self._get("answers/" + ";".join(ids), filter="withbody").get("items", []):
                answers[a["answer_id"]] = a
        results = []
        for q in questions:
            a = answers.get(q.get("accepted_answer_id"))
            if not a:
                continue
            text, codes = html_to_text(a.get("body", ""))
            results.append({
                "title": html.unescape(q.get("title", "")), "question_link": q.get("link"),
                "tags": q.get("tags", []), "score": a.get("score", 0), "answer": text, "code": codes,
                "author": (a.get("owner") or {}).get("display_name", "unknown"),
                "link": f"https://stackoverflow.com/a/{a['answer_id']}", "license": LICENSE})
        return results[:limit]
