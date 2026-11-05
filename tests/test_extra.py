import json
import tempfile
import unittest
from pathlib import Path

from localbrain.extra import offensive_collect, offensive_lessons


class ExtraTests(unittest.TestCase):
    def test_offensive_collect_writes_topic_jsonl(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "offensive"
            (root / "malware").mkdir(parents=True)
            (root / "malware" / "ransomware.md").write_text(
                "# Ransomware\n\n" + "Encrypts user files and demands payment. " * 20, encoding="utf-8")
            (root / "attack-toolkits").mkdir()
            (root / "attack-toolkits" / "metasploit.md").write_text("short", encoding="utf-8")
            output = Path(temp) / "offensive.jsonl"
            counts = offensive_collect(str(root), str(output))
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(counts, {"malware": 1})
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["path"].startswith("offensive/malware/"))
            self.assertIn("Ransomware", rows[0]["text"])

    def test_offensive_collect_requires_existing_root(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                offensive_collect(str(Path(temp) / "missing"), str(Path(temp) / "out.jsonl"))

    def test_offensive_lessons_are_teaching_pairs(self):
        lessons = list(offensive_lessons())
        self.assertGreaterEqual(len(lessons), 5)
        for row in lessons:
            self.assertIn("question", row)
            self.assertIn("answer", row)
            self.assertTrue(row["question"].strip())
            self.assertTrue(row["answer"].strip())
            self.assertLessEqual(len(row["question"]) + len(row["answer"]), 12000)
            self.assertEqual(row.get("source"), "offensive-lesson")


if __name__ == "__main__":
    unittest.main()
