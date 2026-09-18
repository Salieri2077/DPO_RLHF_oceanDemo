import json
import tempfile
import unittest
from pathlib import Path

from evaluator import bootstrap_mean_delta, load_questions, summarize


class EvaluationTest(unittest.TestCase):
    def test_fixed_questions_and_paired_summary(self):
        conversations = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "ocean question"},
            {"role": "assistant", "content": "reference"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps({"conversations": conversations}) + "\n", encoding="utf-8")
            rows = load_questions(path, 1)
        self.assertEqual(rows[0]["messages"], conversations[:-1])
        rows[0]["judge_scores"] = {"sft": -1.0, "grpo": 1.0}
        summary = summarize(rows, seed=42, repeats=100)
        self.assertEqual(summary["mean_score_delta"], 2.0)
        self.assertEqual(summary["wins"], {"sft": 0, "grpo": 1, "tie": 0})
        self.assertEqual(bootstrap_mean_delta([2.0], 42, 10), [2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
