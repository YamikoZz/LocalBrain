"""Extra knowledge tools for the agent: facts and docs come from lookup, not from weights.

  wikipedia(query)        free Wikipedia REST/search API (reliable facts)
  docs_lookup(name)       offline Python docs via pydoc; MDN links for JavaScript names
  explain_error(message)  short hint for common Python/JavaScript errors (for repair prompts)
  search(query)           cached search: local store -> Wikipedia -> web, with fallback
"""
import hashlib
import json
import pydoc
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "LocalBrain/0.2 (self-trained home-PC assistant)"


def _get_json(url, timeout=15):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read(2 * 1024 * 1024).decode("utf-8"))


def wikipedia(query, sentences=4, language="en"):
    """Best matching Wikipedia article summary. Returns dict(title, summary, url) or raises."""
    if not isinstance(query, str) or not query.strip() or len(query) > 300:
        raise ValueError("query must be 1..300 characters")
    if not re.fullmatch(r"[a-z]{2,3}", language):
        raise ValueError("language must be a wiki code like en or th")
    base = f"https://{language}.wikipedia.org"
    found = _get_json(f"{base}/w/api.php?" + urllib.parse.urlencode(
        {"action": "query", "list": "search", "srsearch": query, "srlimit": 1, "format": "json"}))
    hits = found.get("query", {}).get("search", [])
    if not hits:
        raise ValueError("No Wikipedia article found")
    title = hits[0]["title"]
    page = _get_json(f"{base}/api/rest_v1/page/summary/" + urllib.parse.quote(title.replace(" ", "_")))
    text = page.get("extract", "")
    parts = re.split(r"(?<=[.!?])\s+", text)
    return {"title": page.get("title", title), "summary": " ".join(parts[:sentences]),
            "url": page.get("content_urls", {}).get("desktop", {}).get("page", f"{base}/wiki/{title}")}


JS_DOCS = "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/"


def docs_lookup(name, language="python", max_chars=1500):
    """Documentation for a Python name (offline, from the installed stdlib) or a JS reference link."""
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][\w.]{0,100}", name):
        raise ValueError("name must look like str.split, math.gcd or Array.prototype.map")
    if language == "javascript":
        parts = name.replace(".prototype", "").split(".")
        url = JS_DOCS + "/".join(parts)
        return {"name": name, "language": language, "url": url,
                "doc": f"See MDN reference for {name}: {url}"}
    root = name.split(".")[0]
    from .runner import PY_MODULES
    if root not in PY_MODULES | {"str", "list", "dict", "set", "tuple", "int", "float", "bool",
                                 "len", "range", "sorted", "sum", "min", "max", "enumerate", "zip",
                                 "map", "filter", "abs", "round", "any", "all", "reversed", "print"}:
        raise ValueError(f"Docs are limited to built-ins and safe standard modules, not '{root}'")
    obj = pydoc.locate(name)
    if obj is None:
        raise ValueError(f"No documentation found for {name}")
    text = pydoc.render_doc(obj, renderer=pydoc.plaintext)
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    return {"name": name, "language": "python", "doc": text[:max_chars]}


HINTS = [
    (r"IndexError: list index out of range", "An index is past the end of the list; check loop bounds and empty lists."),
    (r"KeyError", "A dictionary key is missing; use dict.get(key, default) or check `in` first."),
    (r"ZeroDivisionError", "Division by zero; handle the empty or zero case before dividing."),
    (r"TypeError: .*NoneType", "A value is None; the function probably forgot to return something."),
    (r"TypeError: unsupported operand", "Mixed types in an operation, e.g. adding str and int; convert one side."),
    (r"TypeError: .*not callable", "Something that is not a function is being called; check names and parentheses."),
    (r"TypeError: .*positional argument", "Wrong number of arguments; check the function signature."),
    (r"NameError: name '(\w+)'", "A name is used before it is defined; check spelling or define '{0}'."),
    (r"AttributeError: '(\w+)' object has no attribute '(\w+)'", "Type '{0}' has no '{1}'; use the right method for that type."),
    (r"RecursionError", "Recursion never stops; add or fix the base case."),
    (r"SyntaxError|IndentationError", "The code does not parse; check colons, brackets and indentation."),
    (r"timed out", "The code runs too long; look for a loop that never ends."),
    (r"returned (.+), expected (.+)", "Wrong result: got {0} but expected {1}; re-check the logic and edge cases."),
    (r"ReferenceError: (\w+) is not defined", "JavaScript: '{0}' is not defined; declare it with let/const."),
    (r"is not a function", "JavaScript: calling something that is not a function; check method names."),
    (r"is not allowed", "The sandbox blocked this; solve it with plain code, no files, network or imports."),
]


def explain_error(message):
    """Short, plain-English hint for an error message (used in repair prompts)."""
    for pattern, hint in HINTS:
        match = re.search(pattern, message or "")
        if match:
            return hint.format(*[g[:60] for g in match.groups()])
    return "Re-read the task, then check edge cases such as empty input, zero and negative numbers."


class SearchCache:
    """Remembers answers per query so repeated searches cost nothing and work offline."""
    def __init__(self, path, max_age_days=30):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_age = max_age_days * 86400
        conn = sqlite3.connect(self.path)
        try:
            with conn:
                conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT, created REAL)")
        finally:
            conn.close()

    def get(self, key):
        conn = sqlite3.connect(self.path)
        try:
            row = conn.execute("SELECT value, created FROM cache WHERE key = ?", (key,)).fetchone()
        finally:
            conn.close()
        if row and time.time() - row[1] < self.max_age:
            return json.loads(row[0])
        return None

    def put(self, key, value):
        conn = sqlite3.connect(self.path)
        try:
            with conn:
                conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)",
                             (key, json.dumps(value, ensure_ascii=False), time.time()))
        finally:
            conn.close()


def search(query, cache=None, knowledge=None, limit=3):
    """Local knowledge first, then Wikipedia, then the web; first non-empty answer wins."""
    key = hashlib.sha256(" ".join(query.lower().split()).encode()).hexdigest()
    if cache:
        hit = cache.get(key)
        if hit:
            return dict(hit, cached=True)
    attempts, result = [], None
    if knowledge is not None:
        try:
            local = knowledge.search(query, limit)
            if local:
                result = {"source": "local", "results": local}
        except Exception as error:  # a broken local index must not stop the search
            attempts.append(f"local: {error}")
    if result is None:
        try:
            result = {"source": "wikipedia", "results": [wikipedia(query)]}
        except Exception as error:
            attempts.append(f"wikipedia: {error}")
    if result is None:
        try:
            from .webtools import search_web
            web = search_web(query, limit)
            if web:
                result = {"source": "web", "results": web}
        except Exception as error:
            attempts.append(f"web: {error}")
    if result is None:
        return {"source": None, "results": [], "errors": attempts}
    if cache:
        cache.put(key, result)
    return dict(result, cached=False)
