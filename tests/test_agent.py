import json
import importlib.util
import math
import unittest
from pathlib import Path

import torch
from transformers import AutoTokenizer
from agent.ocean import (OceanTools, calculate, demonstration, numeric_answer, observation,
                         parse_call, read_jsonl, run_trajectory)
from dataset.lm_dataset import SFTDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.train_agent import AgentSFTDataset, aggregate, round_logps
from trainer.train_grpo import grpo_objective

ROOT = Path(__file__).resolve().parents[1]


class AgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
        cls.tasks = read_jsonl(ROOT / "data/processed/ocean_agent_train.jsonl")
        cls.env = OceanTools(read_jsonl(ROOT / "data/processed/ocean_agent_corpus_train.jsonl"))

    def scripted(self, texts):
        iterator = iter(texts)
        def generate(ids, limit):
            tokens = self.tokenizer.encode(next(iterator), add_special_tokens=False) + [self.tokenizer.eos_token_id]
            self.assertLessEqual(len(tokens), limit)
            return tokens, [-1.] * len(tokens)
        return generate

    def test_units_and_untrusted_calls(self):
        self.assertEqual(calculate(dict(operation="convert", value=1, from_unit="nmi", to_unit="km")), {"value": 1.852, "unit": "km"})
        self.assertEqual(calculate(dict(operation="time", distance=60, speed=20)), {"value": 3, "unit": "h"})
        self.assertEqual(calculate(dict(operation="speed", distance=60, hours=3)), {"value": 20, "unit": "km/h"})
        for args in (dict(operation="convert", value=1, from_unit="km", to_unit="kn"), dict(operation="time", distance=1, speed=0),
                     dict(operation="distance", speed=True, hours=1), dict(operation="distance", speed=float("nan"), hours=1)):
            self.assertIn("error", self.env.execute({"name": "marine_calculate", "arguments": args}))
        self.assertIn("error", self.env.execute({"name": "exec", "arguments": {"code": "print(1)"}}))
        for content in ('<tool_call>null</tool_call>', '<tool_call>{}', '<tool_call>{}</tool_call>'):
            with self.assertRaises(ValueError):
                parse_call(content)
        self.assertTrue(numeric_answer("1.852 km", {"value": 1.852, "unit": "km"}))
        self.assertFalse(numeric_answer("1.852 kn", {"value": 1.852, "unit": "km"}))

    def test_split_isolation_and_demonstrations(self):
        seen_parameters, seen_docs, seen_questions = set(), set(), set()
        for split, count in (("train", 2000), ("val", 50), ("test", 50)):
            tasks = read_jsonl(ROOT / f"data/processed/ocean_agent_{split}.jsonl")
            docs = read_jsonl(ROOT / f"data/processed/ocean_agent_corpus_{split}.jsonl")
            env = OceanTools(docs)
            self.assertEqual(len(tasks), count)
            params = {t["parameter_key"] for t in tasks if "parameter_key" in t}
            sources = {d["id"] for d in docs}
            questions = {t["question"] for t in tasks}
            self.assertFalse(params & seen_parameters)
            self.assertFalse(sources & seen_docs)
            self.assertFalse(questions & seen_questions)
            seen_parameters.update(params); seen_docs.update(sources); seen_questions.update(questions)
            for kind in ("retrieve", "calculate", "chain"):
                task = next(t for t in tasks if t["kind"] == kind)
                demo = demonstration(task, env, self.tokenizer)
                texts = [m["content"] for m in demo["conversations"] if m["role"] == "assistant"]
                trace = run_trajectory(task, self.tokenizer, env, self.scripted(texts))
                self.assertEqual(trace["metrics"]["reward"], 1., trace)
                self.assertEqual(trace["metrics"]["success"], 1.)
                self.assertLessEqual(len(trace["calls"]), 2)

    def test_limits_recovery_and_no_guessing(self):
        task = next(t for t in self.tasks if t["kind"] == "calculate")
        demo = demonstration(task, self.env, self.tokenizer)
        texts = [m["content"] for m in demo["conversations"] if m["role"] == "assistant"]
        guessed = run_trajectory(task, self.tokenizer, self.env, self.scripted([texts[-1]]))
        self.assertEqual(guessed["metrics"]["success"], 0)
        recovered = run_trajectory(task, self.tokenizer, self.env, self.scripted(['<tool_call>{}', *texts]))
        self.assertEqual(recovered["metrics"]["success"], 1)
        limited = run_trajectory(task, self.tokenizer, self.env, self.scripted(texts), max_total_len=10)
        self.assertEqual(limited["stop"], "context_limit")
        self.assertEqual(limited["metrics"]["reward"], -1)
        turn_limit = run_trajectory(task, self.tokenizer, self.env, self.scripted(texts), max_turns=1)
        self.assertEqual(turn_limit["stop"], "turn_limit")
        search = next(t for t in self.tasks if t["kind"] == "retrieve")
        observed = observation(self.env.execute({"name": "search_ocean", "arguments": {"query": search["query"]}}), self.tokenizer)
        self.assertLessEqual(len(self.tokenizer.encode(observed, add_special_tokens=False)), 256)
        self.assertNotIn("error", json.loads(observed))

    def test_multiturn_loss_and_sft_masks(self):
        task = next(t for t in self.tasks if t["kind"] == "chain")
        texts = [m["content"] for m in demonstration(task, self.env, self.tokenizer)["conversations"] if m["role"] == "assistant"]
        trace = run_trajectory(task, self.tokenizer, self.env, self.scripted(texts))
        self.assertEqual(len(trace["rounds"]), 3)
        model = MiniMindForCausalLM(MiniMindConfig(hidden_size=64, num_hidden_layers=1))
        total = sum(len(r["completion_ids"]) for r in trace["rounds"])
        for r in trace["rounds"]:
            self.assertEqual(r["completion_ids"][-1], self.tokenizer.eos_token_id)
            new = round_logps(model, r, torch.device("cpu"))
            self.assertEqual(new.numel(), len(r["completion_ids"]))
            new.retain_grad()
            objective = grpo_objective(new, new.detach(), new.detach(), torch.ones_like(new), torch.ones(1))
            loss = objective["grpo_loss"] * new.numel() / total
            self.assertTrue(math.isfinite(loss.item()))
            loss.backward()
            self.assertGreater(new.grad.abs().sum().item(), 0)
        ds = AgentSFTDataset(ROOT / "data/processed/ocean_agent_sft_train.jsonl", self.tokenizer, 2048, deterministic=True)
        index = self.tasks.index(task)
        ids, labels = ds[index]
        text = self.tokenizer.decode(ids[labels != -100])
        self.assertIn("marine_calculate", text)
        self.assertNotIn("<tool_response>", text)
        # Every real rollout prefix must be the same token sequence seen in SFT.
        # In particular, dropping the empty think block shifts tool-call openings.
        for round_ in trace["rounds"]:
            prefix = round_["input_ids"]
            self.assertEqual(ids[:len(prefix)].tolist(), prefix)
            self.assertEqual(labels[len(prefix)].item(), round_["completion_ids"][0])
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))
        self.assertFalse(aggregate([trace])["gate_pass"])  # one sample has no group signal

    def test_judge_sees_partial_generation(self):
        spec = importlib.util.spec_from_file_location("agent_judge", ROOT / "Agent Evaluation/evaluate_judge.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.candidate_text({"final": "", "rounds": [{"text": "unfinished response"}]}), "unfinished response")
        self.assertEqual(module.candidate_text({"final": "answer", "rounds": [{"text": "tool call"}]}), "answer")
        self.assertEqual(module.candidate_text({"final": "", "rounds": []}), "")


if __name__ == "__main__":
    unittest.main()
