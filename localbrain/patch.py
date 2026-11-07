"""Apply model-written edits to a project (Agent step 5).

Edit format (easy for a small model: no line numbers to get wrong):

    path/to/file.py
    <<<<<<< SEARCH
    exact lines currently in the file
    =======
    replacement lines
    >>>>>>> REPLACE

* Several blocks per patch, several files per patch.
* An empty SEARCH on a file that does not exist creates it.
* Matching: exact first, then ignoring trailing spaces, then ignoring indentation
  (the replacement is re-indented to the matched block). The match must be unique.
* All-or-nothing: if any block fails, no file is touched. A successful apply returns a
  backup so the agent can roll back when tests get worse.
* Paths must stay inside the project (no absolute paths, "..", or .git).
"""
import difflib
import re
from pathlib import Path, PurePosixPath

BLOCK = re.compile(
    r"^(?P<path>[^\n<>=]+?)[ \t]*\n<<<<<<< SEARCH[ \t]*\n(?P<search>.*?)^=======[ \t]*\n"
    r"(?P<replace>.*?)^>>>>>>> REPLACE[ \t]*$", re.S | re.M)
MAX_FILE_BYTES = 2_000_000


class PatchError(ValueError):
    pass


def parse(text):
    """Patch text -> [{"path", "search", "replace"}]. Tolerates ``` fences around blocks."""
    text = re.sub(r"^```[\w-]*[ \t]*$", "", text.replace("\r\n", "\n"), flags=re.M)
    blocks = [{"path": m["path"].strip().strip("`*"), "search": m["search"], "replace": m["replace"]}
              for m in BLOCK.finditer(text)]
    if not blocks:
        raise PatchError("no SEARCH/REPLACE blocks found")
    return blocks


def safe_path(root, rel):
    rel = rel.replace("\\", "/").strip()
    parts = PurePosixPath(rel).parts
    if not rel or rel.startswith("/") or re.match(r"^[A-Za-z]:", rel) or ".." in parts or ".git" in parts:
        raise PatchError(f"path not allowed: {rel!r}")
    full = (Path(root) / rel).resolve()
    if Path(root).resolve() not in full.parents:
        raise PatchError(f"path escapes project: {rel!r}")
    return full


def _find(content, search):
    """Return (start, end, matched_text) for a unique match, trying looser strategies."""
    # Exact match, but only at line starts (an unindented SEARCH must not match mid-line).
    starts = [m.start() for m in re.finditer(re.escape(search), content)
              if m.start() == 0 or content[m.start() - 1] == "\n"]
    if len(starts) == 1:
        return starts[0], starts[0] + len(search), search
    if len(starts) > 1:
        raise PatchError("SEARCH text matches more than once; include more surrounding lines")
    lines = content.splitlines(keepends=True)
    want = search.splitlines()
    for norm in (lambda s: s.rstrip(), lambda s: s.strip()):
        target = [norm(l) for l in want]
        hits = [i for i in range(len(lines) - len(want) + 1)
                if [norm(l) for l in lines[i:i + len(want)]] == target]
        if len(hits) > 1:
            raise PatchError("SEARCH text matches more than once; include more surrounding lines")
        if hits:
            start = sum(len(l) for l in lines[:hits[0]])
            matched = "".join(lines[hits[0]:hits[0] + len(want)])
            return start, start + len(matched), matched
    raise PatchError("SEARCH text not found:\n" + "\n".join(want[:5]))


def _reindent(replace, search, matched):
    """If the match only differed by indentation, shift the replacement by the same amount."""
    def indent(text):
        first = next((l for l in text.splitlines() if l.strip()), "")
        return first[:len(first) - len(first.lstrip())]
    have, want = indent(matched), indent(search)
    if have == want:
        return replace
    out = []
    for line in replace.splitlines(keepends=True):
        if line.strip() and line.startswith(want):
            line = have + line[len(want):]
        elif line.strip() and not want:
            line = have + line
        out.append(line)
    return "".join(out)


def apply_patch(root, text, dry_run=False):
    """Apply a patch under root. Returns {"files", "diff", "backup"}; raises PatchError on failure."""
    root = Path(root)
    new = {}       # path -> new content
    original = {}  # path -> old content (None = file did not exist)
    for n, block in enumerate(parse(text), 1):
        path = safe_path(root, block["path"])
        if path not in new:
            if path.exists():
                if path.stat().st_size > MAX_FILE_BYTES:
                    raise PatchError(f"{block['path']}: file too large")
                original[path] = new[path] = path.read_text(encoding="utf-8")
            else:
                original[path], new[path] = None, None
        content = new[path]
        try:
            if content is None:
                if block["search"].strip():
                    raise PatchError("file does not exist (use an empty SEARCH to create it)")
                new[path] = block["replace"]
                continue
            if not block["search"].strip():
                raise PatchError("empty SEARCH on an existing file; give the lines to replace")
            start, end, matched = _find(content, block["search"])
            replace = _reindent(block["replace"], block["search"], matched)
            if matched.endswith("\n") and replace and not replace.endswith("\n"):
                replace += "\n"
            new[path] = content[:start] + replace + content[end:]
        except PatchError as error:
            raise PatchError(f"block {n} ({block['path']}): {error}") from None
    diff = []
    for path, content in new.items():
        rel = path.relative_to(root.resolve()).as_posix()
        diff += difflib.unified_diff((original[path] or "").splitlines(keepends=True),
                                     content.splitlines(keepends=True), f"a/{rel}", f"b/{rel}")
    if not dry_run:
        for path, content in new.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="")
    return {"files": sorted(p.relative_to(root.resolve()).as_posix() for p in new),
            "diff": "".join(diff), "backup": {str(p): c for p, c in original.items()},
            "applied": not dry_run}


def revert(backup):
    """Undo apply_patch using its backup (restores old content, deletes created files)."""
    for path, content in backup.items():
        path = Path(path)
        if content is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(content, encoding="utf-8", newline="")
