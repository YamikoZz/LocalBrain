"""Download data files only. No model, inference API, or remote code execution."""
import hashlib
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

COMMIT = "99c6aeda56b2dd9d0f25217208bc2560254d1bf2"
URL = ("https://raw.githubusercontent.com/wannaphong/han-instruction-dataset/"
       + COMMIT + "/han-instruction-dataset.json")


def main():
    root = Path(__file__).resolve().parents[1]
    target = root / "workspace/raw/han-instruction-dataset.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temp = target.with_suffix(".download")
        try:
            with urllib.request.urlopen(URL, timeout=60) as response, temp.open("wb") as stream:
                size = 0
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > 128 * 1024 * 1024:
                        raise ValueError("Dataset exceeds 128 MiB limit")
                    stream.write(chunk)
            json.loads(temp.read_text(encoding="utf-8"))
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = {"dataset": "Han Instruction Dataset", "author": "Wannaphong Phatthiyaphaibun / PyThaiNLP",
                "repository": "https://github.com/wannaphong/han-instruction-dataset",
                "url": URL, "commit": COMMIT, "sha256": digest,
                "bytes": target.stat().st_size, "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "license": "CC-BY-SA-4.0 (dataset README; repository code is Apache-2.0)",
                "origin": "Existing human and synthetic examples; no model/API called to generate data",
                "changes": "Raw download unchanged; training preparation deduplicates and filters by length"}
    (target.parent / "SOURCE.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
