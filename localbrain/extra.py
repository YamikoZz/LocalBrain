"""Extra domain data: FiveM Lua and cybersecurity, written as path/text JSONL.

FiveM: Lua resources from MIT / GPL / LGPL repositories (personal-use training only).
Security (defensive / educational):
  * OWASP Cheat Sheet Series and Web Security Testing Guide (CC-BY-SA-4.0)
  * MITRE CWE weakness list (descriptions, mitigations, vulnerable-vs-fixed examples)
  * security-themed FineWeb-Edu articles already on disk
Also builds CWE question/answer pairs for SFT.

Scope: defensive and educational only (how weaknesses work and how to prevent them).
No malware source, exploit code, or attack-tooling collection lives here by design.
"""
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

FIVEM_LICENSES = {"mit", "gpl-3.0", "lgpl-3.0"}


def fivem_collect(root, output):
    root, rows = Path(root), 0
    with Path(output).open("w", encoding="utf-8") as out:
        for repo in sorted(p for p in root.iterdir() if p.is_dir()):
            for path in sorted(repo.rglob("*.lua")):
                if ".git" in path.parts or path.stat().st_size > 200_000:
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                if len(text.strip()) < 60:
                    continue
                name = repo.name.replace("__", "/")
                out.write(json.dumps({"path": f"FiveM/{name}/{path.relative_to(repo).as_posix()}",
                                      "text": f"-- platform: FiveM (GTA V, CfxLua 5.4)\n{text}"},
                                     ensure_ascii=False) + "\n")
                rows += 1
    return rows


def _text(element):
    """Flatten CWE XHTML-ish structured text to plain text with code blocks kept."""
    if element is None:
        return ""
    parts = []
    for node in element.iter():
        tag = node.tag.split("}")[-1]
        if tag in {"Example_Code", "code"}:
            code = "".join(node.itertext()).strip("\n")
            parts.append("\n```\n" + code + "\n```\n")
            node.clear()
        elif node.text and node.text.strip():
            parts.append(node.text.strip())
        if node.tail and node.tail.strip():
            parts.append(node.tail.strip())
    return re.sub(r"[ \t]+", " ", " ".join(parts)).strip()


def cwe_entries(xml_path):
    tree = ET.parse(xml_path)
    ns = {"c": tree.getroot().tag.split("}")[0].strip("{")}
    for weakness in tree.getroot().iterfind(".//c:Weakness", ns):
        if weakness.get("Status") == "Deprecated":
            continue
        cid, name = weakness.get("ID"), weakness.get("Name")
        desc = _text(weakness.find("c:Description", ns))
        extended = _text(weakness.find("c:Extended_Description", ns))
        mitigations = [_text(m.find("c:Description", ns)) for m in
                       weakness.iterfind("c:Potential_Mitigations/c:Mitigation", ns)]
        examples = [_text(e) for e in weakness.iterfind("c:Demonstrative_Examples/c:Demonstrative_Example", ns)]
        yield {"id": f"CWE-{cid}", "name": name, "description": desc, "extended": extended,
               "mitigations": [m for m in mitigations if m], "examples": [e for e in examples if e]}


def security_collect(security_root, fineweb_files, output, fineweb_min_hits=3):
    import pyarrow.parquet as pq
    root, counts = Path(security_root), {}
    keywords = re.compile(r"\b(cybersecurity|cyber security|encryption|vulnerabilit|phishing|firewall|malware|"
                          r"authentication|cryptograph|SQL injection|cross-site|penetration test|"
                          r"information security|network security)", re.I)
    with Path(output).open("w", encoding="utf-8") as out:
        def write(kind, path, text):
            out.write(json.dumps({"path": path, "text": text}, ensure_ascii=False) + "\n")
            counts[kind] = counts.get(kind, 0) + 1

        for folder, kind in ((root / "CheatSheetSeries/cheatsheets", "owasp-cheatsheets"),
                             (root / "wstg/document", "owasp-wstg")):
            for md in sorted(folder.rglob("*.md")):
                text = md.read_text(encoding="utf-8", errors="replace")
                if len(text) > 200:
                    write(kind, f"security/{kind}/{md.name}", text)
        for xml in sorted(root.glob("cwec_*.xml")):
            for e in cwe_entries(xml):
                body = [f"# {e['id']}: {e['name']}", e["description"], e["extended"]]
                if e["mitigations"]:
                    body.append("## Mitigations\n" + "\n".join("- " + m for m in e["mitigations"]))
                if e["examples"]:
                    body.append("## Examples\n" + "\n\n".join(e["examples"][:3]))
                write("cwe", f"security/cwe/{e['id']}.md", "\n\n".join(b for b in body if b))
        for path in fineweb_files:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=["text", "int_score"]):
                for row in batch.to_pylist():
                    if row["int_score"] >= 3 and len(keywords.findall(row["text"][:3000])) >= fineweb_min_hits:
                        write("fineweb-security", "security/articles/article.md", row["text"])
    return counts


def cwe_pairs(security_root):
    """SFT pairs: explain a weakness, how to prevent it, and spot it in code."""
    for xml in sorted(Path(security_root).glob("cwec_*.xml")):
        for e in cwe_entries(xml):
            if not e["description"]:
                continue
            short = e["description"].split(". ")[0].rstrip(".") + "."
            yield {"question": f"What is {e['id']} ({e['name']})?",
                   "answer": f"{e['id']}, {e['name']}, is a software weakness. {e['description']}",
                   "source": "security-cwe"}
            if e["mitigations"]:
                tips = " ".join(e["mitigations"][:3])
                yield {"question": f"How can I prevent {e['name'].lower()} in my code?",
                       "answer": f"{short} To prevent it: {tips}", "source": "security-cwe"}
