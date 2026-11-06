"""Extra domain data: FiveM Lua and cybersecurity, written as path/text JSONL.

FiveM: Lua resources from MIT / GPL / LGPL repositories (personal-use training only).
Security (defensive / educational):
  * OWASP Cheat Sheet Series and Web Security Testing Guide (CC-BY-SA-4.0)
  * MITRE CWE weakness list (descriptions, mitigations, vulnerable-vs-fixed examples)
  * security-themed FineWeb-Edu articles already on disk
Also builds CWE question/answer pairs for SFT.
Offensive security (malware / attack tooling / training lessons):
  * malware analysis documents (static + dynamic analysis notes, RE write-ups)
  * attack toolkit and red-team / adversary knowledge (C2, exploits, TTPs)
  * hand-written teaching pairs for malware analysis and offensive tooling
"""
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

FIVEM_LICENSES = {"mit", "gpl-3.0", "lgpl-3.0"}


def offensive_collect(root, output, min_chars=60):
    """Collect offensive-security documents (malware analysis, ATT&CK/TTP notes, attack
    toolkits, red-team guides) as path/text JSONL.

    Each immediate subdirectory of ``root`` is treated as one topic kind
    (e.g. malware, attack-toolkits, red-team, c2, threat-intel) and every ``*.md``
    below it becomes one training document. Paths keep the ``offensive/<kind>/...``
    shape so the annealing mix can name them like the security extras.
    """
    root, counts = Path(root), {}
    if not root.is_dir():
        raise ValueError(f"Offensive security root not found: {root}")
    with Path(output).open("w", encoding="utf-8") as out:
        def write(kind, path, text):
            out.write(json.dumps({"path": path, "text": text}, ensure_ascii=False) + "\n")
            counts[kind] = counts.get(kind, 0) + 1

        for folder, kind in ((p, p.name) for p in sorted(root.iterdir()) if p.is_dir()):
            for md in sorted(folder.rglob("*.md")):
                text = md.read_text(encoding="utf-8", errors="replace")
                if len(text.strip()) < min_chars:
                    continue
                write(kind, f"offensive/{md.relative_to(root).as_posix()}", text)
    return counts


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


def offensive_lessons():
    """Hand-written teaching pairs: malware analysis and offensive tooling knowledge."""
    yield {"question": "What is malware analysis?",
           "answer": "Malware analysis is the work of understanding what a malicious program does "
                      "before defenders can detect or remove it. Analysts usually begin with static "
                      "analysis: file type, cryptographic hash, entropy, byte patterns, printable "
                      "strings and disassembly. When static analysis is not enough they continue with "
                      "dynamic analysis by running the sample in a controlled sandbox and watching "
                      "processes, file system changes, network traffic and persistence mechanisms. "
                      "The findings become indicators of compromise and inputs for detection and "
                      "mitigation rules.",
           "source": "offensive-lesson"}
    yield {"question": "What is the difference between a vulnerability, an exploit and malware?",
           "answer": "A vulnerability is a weakness in software or configuration. An exploit is a piece "
                      "of code or technique that uses that weakness. Malware is the malicious payload "
                      "or code delivered through an exploit or other infection vector. Defenders reduce "
                      "risk by fixing vulnerabilities, monitoring for exploits and building malware "
                      "detection and response playbooks.",
           "source": "offensive-lesson"}
    yield {"question": "What is the MITRE ATT&CK framework?",
           "answer": "The MITRE ATT&CK framework is a knowledge base of real-world adversary tactics and "
                      "techniques. Tactics describe the attacker's goal at each stage, techniques describe "
                      "how the goal is achieved, and the framework maps them to platforms and data sources. "
                      "Threat teams use ATT&CK to describe adversary behavior, share intelligence and "
                      "align detection engineering with the techniques being used.",
           "source": "offensive-lesson"}
    yield {"question": "What is a command-and-control server?",
           "answer": "Command-and-control, or C2, is the infrastructure a compromised system uses to "
                      "receive instructions from the attacker and exfiltrate data. C2 traffic can be direct "
                      "or through a chain of proxies, and defenders detect it through network signatures, "
                      "domain and IP reputation, TLS fingerprinting and behavioral monitoring.",
           "source": "offensive-lesson"}
    yield {"question": "What is a zero-day exploit?",
           "answer": "A zero-day exploit targets a vulnerability that the vendor has not patched yet, so "
                      "defenders have no defense available before the fix ships. Zero-days are expensive and "
                      "rare in real operations, but they are the most dangerous class of offensive capability "
                      "because both the vulnerability and the exploit are unknown ahead of time.",
           "source": "offensive-lesson"}
    yield {"question": "How do penetration testers use the Metasploit Framework?",
           "answer": "The Metasploit Framework is an open-source platform of offensive security modules. "
                      "Testers select a target and a module, set the required options such as the local "
                      "exploit and the payload, and run it inside an authorized lab or approved target. "
                      "The framework includes auxiliary, exploit and post-exploitation modules, and its "
                      "value is the repeatable workflow: pick technique, configure, execute, document.",
           "source": "offensive-lesson"}
    yield {"question": "What is a reverse shell?",
           "answer": "A reverse shell is a payload that makes an attacked machine dial back to the attacker "
                      "instead of listening on a port the attacker must reach. It is a common technique "
                      "because outbound connections are usually allowed where inbound ones are filtered, "
                      "and defenders hunt for the beaconing pattern and the parent process relationship.",
           "source": "offensive-lesson"}
    yield {"question": "What is threat modeling?",
           "answer": "Threat modeling is the proactive practice of identifying which threats could affect a "
                      "system and how to reduce them before code or infrastructure ships. Approaches such as "
                      "STRIDE decompose threats into spoofing, tampering, repudiation, information disclosure, "
                      "denial of service and elevation of privilege, then map mitigations like threat models, "
                      "input validation and least privilege.",
           "source": "offensive-lesson"}
