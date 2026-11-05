from pathlib import Path

from .webtools import html_text


def read_document(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Document must be a file no larger than 8 MiB")
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise ValueError("PDF requires optional dependency: pip install -e .[pdf]") from error
        text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    elif path.suffix.lower() in {".txt", ".md", ".csv", ".html", ".htm", ".py", ".json"}:
        text = path.read_text(encoding="utf-8-sig")
        if path.suffix.lower() in {".html", ".htm"}:
            text, _ = html_text(text)
    else:
        raise ValueError("Supported: TXT, MD, CSV, HTML, PDF, PY, JSON")
    if len(text) > 4_000_000:
        raise ValueError("Extracted document too large")
    return {"text": text, "source": path.resolve().as_uri(), "title": path.name}
