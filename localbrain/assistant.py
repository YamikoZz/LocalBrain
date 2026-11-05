import time
from pathlib import Path

from .documents import read_document
from .knowledge import Knowledge
from .webtools import read_webpage, search_web


def ingest(knowledge, source):
    if source.startswith(("http://", "https://")):
        documents = [read_webpage(source)]
    else:
        path = Path(source)
        paths = sorted(p for p in path.rglob("*")
                       if p.is_file() and p.suffix.lower() in
                       {".txt", ".md", ".csv", ".html", ".htm", ".pdf"}) if path.is_dir() else [path]
        if not paths:
            raise ValueError("No supported documents found")
        documents = (read_document(p) for p in paths)
    results = []
    for document in documents:
        results.append({"source": document["source"],
                        **knowledge.add(document["text"], document["source"], document["title"])})
    return results


def research(knowledge, query, limit=3):
    results = []
    for item in search_web(query, limit):
        try:
            page = read_webpage(item["url"])
            result = knowledge.add(page["text"], page["source"], page["title"])
            results.append({"source": page["source"], **result})
        except (ValueError, OSError) as error:
            results.append({"source": item["url"], "error": str(error)})
        time.sleep(1)
    return results


def refresh(knowledge):
    results = []
    # Refresh only already stored public web sources; local files are re-ingested explicitly.
    for item in knowledge.sources():
        url = item["source"]
        if not url.startswith(("http://", "https://")):
            continue
        try:
            page = read_webpage(url)
            result = knowledge.add(page["text"], url, page["title"])
            results.append({"source": url, **result})
        except (ValueError, OSError) as error:
            results.append({"source": url, "error": str(error)})
        time.sleep(1)
    return results


def answer(knowledge, query, run=None, use_web=False, device="auto"):
    greeting = query.strip().casefold().rstrip("!?.。！ ")
    if greeting in {"hello", "hi", "hey", "สวัสดี", "สวัสดีครับ", "สวัสดีค่ะ", "หวัดดี"}:
        return {"mode": "greeting",
                "answer": "สวัสดีครับ อยากให้ช่วยค้นเรื่องอะไรครับ?",
                "sources": []}
    hits = knowledge.search(query, 3)
    if not hits and use_web:
        research(knowledge, query)
        hits = knowledge.search(query, 3)
    if not hits:
        return {"mode": "no_evidence",
                "answer": "ยังไม่พบหลักฐานในคลัง ลองเพิ่มเอกสารหรือใช้ --web เพื่อค้นเพิ่ม",
                "sources": []}
    sources = [{"id": i, "source": item["source"], "title": item["title"],
                "fetched_at": item["fetched_at"]} for i, item in enumerate(hits, 1)]
    if not run:
        return {"mode": "retrieved_excerpts",
                "answer": "\n\n".join(f"[{i}] {item['body']}" for i, item in enumerate(hits, 1)),
                "sources": sources,
                "note": "ข้อความคัดจากเอกสารตามผลค้น ไม่ใช่คำตอบที่โมเดลตรวจสอบแล้ว"}
    import json
    from tokenizers import Tokenizer
    from .train import generate
    run = Path(run)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    tok = Tokenizer.from_file(str(run / "tokenizer.json"))
    # Context is short in this prototype. Budget explicitly rather than silently
    # pretending the model can consume an entire retrieved document.
    suffix = "\nUser: " + query + "\nAssistant:"
    suffix_ids = tok.encode(suffix).ids
    budget = cfg["context"] - len(suffix_ids) - 16
    if budget < 24:
        raise ValueError("Question too long for this model's context")
    evidence = "\n".join(f"[{i}] {h['body']}" for i, h in enumerate(hits, 1))
    evidence = tok.decode(tok.encode(evidence).ids[:budget])
    prompt = "Context:\n" + evidence + suffix
    return {"mode": "experimental_model",
            "answer": generate(run, prompt, max_tokens=100, device=device),
            "sources": sources,
            "note": "ผลจากโมเดลทดลอง อาจผิดหรืออ้างอิงไม่ตรง ควรตรวจข้อความต้นฉบับ"}
