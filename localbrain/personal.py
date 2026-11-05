"""Collect the owner's own code for private training (never uploaded; lives under workspace/).

Secrets are removed before anything is written: API keys, tokens, private keys and
password/secret assignments are replaced with <REDACTED>. Files that are usually
secret (.env, key files) are skipped entirely.
"""
import hashlib
import json
import re
from pathlib import Path

CODE = {".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".lua", ".luau", ".html", ".htm", ".css",
        ".scss", ".php", ".c", ".h", ".cpp", ".hpp", ".cs", ".java", ".go", ".rs", ".sh", ".ps1", ".sql"}
JSON_OK = {"minecraft", "REA", "Mcaddon"}  # addon/game definitions are code-like; other JSON is data
SKIP_DIRS = {"node_modules", ".git", ".venv", "venv", "site-packages", "__pycache__", "dist", "build",
             ".next", ".cache", "vendor", "coverage"}
# Decompiled third-party apps are not the owner's code (and are obfuscated): never collect them.
SKIP_DIR_PATTERN = re.compile(r"(-src$|^sources$|decompiled?|jadx|apktool)", re.I)
SKIP_NAMES = re.compile(r"(^\.env|secret|credential|private[_-]?key|\.pem$|\.key$|id_rsa|token)", re.I)

SECRETS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"\b(sk-[A-Za-z0-9_\-]{16,}|sk-ant-[A-Za-z0-9_\-]{16,})"),        # OpenAI / Anthropic style
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"),                                    # Google API key
    re.compile(r"\b(ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{20,}"),             # GitHub tokens
    re.compile(r"\bxox[abpr]-[A-Za-z0-9\-]{10,}"),                               # Slack tokens
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                                         # AWS access key id
    re.compile(r"\b[MN][A-Za-z\d]{23,25}\.[\w-]{6}\.[\w-]{27,}"),                # Discord bot token
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),  # JWT
]
ASSIGN = re.compile(r"""(?ix)
    \b([\w-]*(?:password|passwd|pwd|secret|api[_-]?key|apikey|access[_-]?token|auth[_-]?token|
             client[_-]?secret|private[_-]?key|webhook)[\w-]*)
    (\s*[:=]\s*|\s*=>\s*|\(\s*['"][^'"]*['"]\s*,\s*)
    (['"])([^'"\n]{4,})(['"])""")


def redact(text):
    count = 0
    for pattern in SECRETS:
        text, n = pattern.subn("<REDACTED>", text)
        count += n

    def mask(m):
        nonlocal count
        value = m.group(4)
        if value in {"<REDACTED>", "your-api-key", "YOUR_API_KEY", "changeme", "password"} or value.startswith("$"):
            return m.group(0)
        count += 1
        return m.group(1) + m.group(2) + m.group(3) + "<REDACTED>" + m.group(5)

    text = ASSIGN.sub(mask, text)
    return text, count


def collect(folders, output, max_bytes=200_000):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen, stats = set(), {"files": 0, "duplicates": 0, "skipped_secret_files": 0, "redactions": 0,
                          "by_extension": {}, "by_folder": {}}
    with output.open("w", encoding="utf-8") as stream:
        for folder in folders:
            root = Path(folder)
            for path in sorted(root.rglob("*")):
                if not path.is_file() or any(part in SKIP_DIRS or SKIP_DIR_PATTERN.search(part)
                                             for part in path.relative_to(root).parts[:-1]):
                    continue
                ext = path.suffix.lower()
                if ext not in CODE and not (ext == ".json" and root.name in JSON_OK):
                    continue
                if SKIP_NAMES.search(path.name):
                    stats["skipped_secret_files"] += 1
                    continue
                if path.stat().st_size > max_bytes:
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                if len(text.strip()) < 40:
                    continue
                key = hashlib.sha1(" ".join(text.split()).encode()).hexdigest()
                if key in seen:
                    stats["duplicates"] += 1
                    continue
                seen.add(key)
                text, n = redact(text)
                stats["redactions"] += n
                rel = path.relative_to(root.parent).as_posix()
                stream.write(json.dumps({"path": rel, "text": text}, ensure_ascii=False) + "\n")
                stats["files"] += 1
                stats["by_extension"][ext] = stats["by_extension"].get(ext, 0) + 1
                stats["by_folder"][root.name] = stats["by_folder"].get(root.name, 0) + 1
    return stats


def documents(path):
    """Training documents: a path header (in the file's own comment style) + the code."""
    styles = {".py": "# ", ".sh": "# ", ".ps1": "# ", ".lua": "-- ", ".luau": "-- ", ".sql": "-- ",
              ".html": None, ".htm": None, ".css": None, ".json": None}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        ext = Path(row["path"]).suffix.lower()
        prefix = styles.get(ext, "// ")
        header = f"{prefix}file: {row['path']}\n" if prefix else ""
        yield header + row["text"]
