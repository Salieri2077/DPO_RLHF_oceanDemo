import unittest

from evaluator import aggregate, bootstrap_delta, encoded_candidates
from prepare_benchmarks import adapt_arc, adapt_maritime, stable_rows


class FakeTokenizer:
    bos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 100 + 2 for char in text]


class EvaluationTest(unittest.TestCase):
    def test_stable_sampling_and_adapters(self):
        rows = [{"id": str(index)} for index in range(20)]
        self.assertEqual(stable_rows(rows, 5, lambda row: row["id"]), stable_rows(reversed(rows), 5, lambda row: row["id"]))
        arc = adapt_arc({"id": "x", "question": "Q", "choices": {"label": ["A", "B"], "text": ["one", "two"]}, "answerKey": "B"})
        maritime = adapt_maritime({"question": "Q", "answer": "C", "A": "a", "B": "b", "C": "c", "D": "d"})
        self.assertEqual((arc["answer"], maritime["answer"]), (1, 2))

    def test_encoding_truncates_only_context(self):
        row = {"id": "1", "benchmark": "b", "domain": "d", "prompt": "abcdefgh", "choices": ["x", "yz"],
               "answer": 0, "choice_prefix": " "}
        candidates = encoded_candidates([row], FakeTokenizer(), 6)
        self.assertTrue(all(item["truncated"] for item in candidates))
        self.assertEqual([item["continuation_tokens"] for item in candidates], [2, 3])
        self.assertTrue(all(len(item["ids"]) <= 6 for item in candidates))

    def test_aggregation_and_paired_delta(self):
        rows = [{"benchmark": "b", "domain": "d", "correct": value, "correct_norm": value,
                 "correct_nll": 2.0, "correct_tokens": 2, "context_truncated": False} for value in (1, 0)]
        self.assertEqual(aggregate(rows)["domain"]["d"]["accuracy_norm"], 0.5)
        delta, interval = bootstrap_delta([0, 0, 0], [1, 1, 1], 42, 20)
        self.assertEqual((delta, interval), (1.0, [1.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
