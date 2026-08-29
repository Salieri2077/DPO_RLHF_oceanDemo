import unittest

from evaluator import aggregate_benchmarks, bootstrap_delta, chat_candidates


class FakeTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "<bos>" + "|".join(item["content"] for item in messages) + "|assistant:"

    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 97 + 3 for char in text]


class EvaluationTest(unittest.TestCase):
    def test_chat_encoding_keeps_answer_when_context_is_truncated(self):
        rows = [{"id": "1", "benchmark": "b", "domain": "d", "prompt": "x" * 100,
                 "choices": ["short", "longer"], "choice_prefix": "", "answer": 0}]
        encoded = chat_candidates(rows, FakeTokenizer(), 32)
        self.assertTrue(all(item["truncated"] and len(item["ids"]) == 32 for item in encoded))
        self.assertEqual([item["continuation_tokens"] for item in encoded], [10, 11])

    def test_metrics_and_paired_bootstrap(self):
        rows = [
            {"benchmark": "b", "domain": "d", "correct": 1, "correct_norm": 1,
             "correct_nll": 2.0, "correct_tokens": 2, "context_truncated": False},
            {"benchmark": "b", "domain": "d", "correct": 0, "correct_norm": 0,
             "correct_nll": 4.0, "correct_tokens": 2, "context_truncated": True},
        ]
        self.assertEqual(aggregate_benchmarks(rows)["overall"]["accuracy_norm"], 0.5)
        delta, interval = bootstrap_delta([1, 1, 1], [0, 0, 0], seed=42)
        self.assertEqual((delta, interval), (-1.0, [-1.0, -1.0]))


if __name__ == "__main__":
    unittest.main()
