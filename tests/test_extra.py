import json
import tempfile
import unittest
from pathlib import Path

from localbrain.extra import cwe_pairs, fivem_collect


class ExtraTests(unittest.TestCase):
    def test_fivem_collect_tags_platform(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "fivem"
            (root / "qb-core").mkdir(parents=True)
            (root / "qb-core" / "main.lua").write_text(
                "RegisterNetEvent('qb:server:test')\n" + "local x = 1\n" * 20, encoding="utf-8")
            (root / "qb-core" / "tiny.lua").write_text("x=1", encoding="utf-8")
            output = Path(temp) / "fivem.jsonl"
            rows = fivem_collect(str(root), str(output))
            docs = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(rows, 1)  # the tiny file is skipped
            self.assertTrue(docs[0]["path"].startswith("FiveM/qb-core/"))
            self.assertIn("platform: FiveM", docs[0]["text"])


if __name__ == "__main__":
    unittest.main()
