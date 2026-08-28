import math
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.rollout_engine import rollout
from trainer.train_grpo import grpo_objective, group_advantages, repetition_penalty
from trainer.trainer_utils import SiliconFlowRewardModel, parse_reward_group


class GRPOTest(unittest.TestCase):
    def test_torch_rollout_policy_backward(self):
        model = MiniMindForCausalLM(MiniMindConfig(hidden_size=64, num_hidden_layers=1))
        tokenizer = AutoTokenizer.from_pretrained(Path(__file__).resolve().parents[1] / "model")
        prompt = tokenizer("海洋", return_tensors="pt", add_special_tokens=False)
        result = rollout(model, tokenizer, prompt["input_ids"], prompt["attention_mask"], 2, 2)
        output = model(result.output_ids, logits_to_keep=result.completion_ids.size(1) + 1)
        logps = torch.log_softmax(output.logits[:, :-1], -1).gather(2, result.completion_ids.unsqueeze(-1)).squeeze(-1)
        metrics = grpo_objective(logps, result.old_log_probs, result.old_log_probs, result.completion_mask, torch.tensor([1.0, -1.0]))
        metrics["grpo_loss"].backward()

    def test_tiny_moe_forward_backward(self):
        config = MiniMindConfig(hidden_size=64, num_hidden_layers=1, use_moe=True)
        model = MiniMindForCausalLM(config)
        inputs = torch.randint(0, config.vocab_size, (1, 8))
        output = model(inputs, labels=inputs)
        loss = output.loss + output.aux_loss
        self.assertTrue(math.isfinite(loss.item()))
        loss.backward()

    def test_advantages_objective_and_penalty(self):
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        advantages, stds = group_advantages(rewards, 4)
        self.assertAlmostEqual(advantages.mean().item(), 0.0, places=5)
        self.assertGreater(stds.item(), 0)
        new = torch.zeros(4, 3, requires_grad=True)
        metrics = grpo_objective(new, torch.zeros_like(new), torch.zeros_like(new), torch.ones_like(new), advantages)
        self.assertTrue(math.isfinite(metrics["grpo_loss"].item()))
        metrics["grpo_loss"].backward()
        self.assertGreater(repetition_penalty("abc abc abc abc"), 0)

    def test_group_judge_json_and_retry(self):
        self.assertEqual(parse_reward_group("```json\n[1, -2]\n```", 2), [1.0, -2.0])
        judge = SiliconFlowRewardModel("test-key")
        replies = iter(["not json", "[1, 2, 3, 3]"])
        judge._request = lambda _: next(replies)
        with patch("trainer.trainer_utils.time.sleep"):
            self.assertEqual(judge.score_group("海浪是什么？", "参考", ["a", "b", "c", "d"]), [1.0, 2.0, 3.0, 3.0])


if __name__ == "__main__":
    unittest.main()
