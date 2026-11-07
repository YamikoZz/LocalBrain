import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import import_code_sft as imp  # noqa: E402


class DecontaminationTests(unittest.TestCase):
    def setUp(self):
        long_task = "Write a function to find the shared elements from the given two lists and return them sorted."
        short_task = "Write a function to reverse words in a string."
        grams = imp.ngrams(long_task)
        short = {" " + " ".join(imp.words(short_task)) + " "}
        self.eval = (grams, short)

    def test_catches_long_and_short_eval_tasks_even_when_wrapped(self):
        self.assertTrue(imp.overlaps_eval(
            "Please help. Write a function to find the shared elements from the given two lists and "
            "return them sorted. Use Python.", self.eval))
        self.assertTrue(imp.overlaps_eval("Task: write a function to reverse words in a string!", self.eval))

    def test_shared_code_boilerplate_is_not_contamination(self):
        self.assertFalse(imp.overlaps_eval(
            "Write a function that counts vowels; loop with for i in range(1, n + 1) over the text.", self.eval))

    def test_fence_language(self):
        self.assertEqual(imp.fence_language("text\n```lua\nprint(1)\n```"), "lua")
        self.assertIsNone(imp.fence_language("no code here"))


if __name__ == "__main__":
    unittest.main()
