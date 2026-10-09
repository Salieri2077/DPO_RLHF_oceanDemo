import math
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.rollout_engine import rollout
from trainer.train_grpo import grpo_objective, group_advantages, repetition_penalty
from trainer.rl_schedule import PlateauController, scheduled_lr
from trainer.trainer_utils import SiliconFlowRewardModel, parse_reward_group

# Evaluations of the 400-step Qwen3-1.7B GRPO run, steps 25..400: validation reward, mean training reward and mean KL
# over the preceding 25 steps.
QWEN_RUN = [(25, 1.73, 1.61, 5.6e-5), (50, 1.85, 1.71, 2.6e-4), (75, 1.73, 1.79, 8.2e-4), (100, 2.09, 2.06, 1.6e-3),
            (125, 2.08, 1.95, 2.5e-3), (150, 2.05, 1.88, 3.1e-3), (175, 2.21, 1.94, 3.7e-3), (200, 2.33, 2.01, 4.2e-3),
            (225, 2.03, 2.04, 5.1e-3), (250, 1.98, 2.05, 6.5e-3), (275, 2.02, 2.15, 9.3e-3), (300, 2.13, 2.00, 1.2e-2),
            (325, 2.19, 1.89, 1.1e-2), (350, 2.14, 2.07, 1.2e-2), (375, 2.20, 2.03, 1.2e-2), (400, 2.06, 2.13, 1.1e-2)]


def replay(controller):
    for step, val, train, kl in QWEN_RUN:
        controller.update(step, val, train, kl)
        if controller.decay_start is not None:
            return controller.decay_start
    return None


class RLScheduleTest(unittest.TestCase):
    def test_wsd_rewarms_holds_and_decays(self):
        lr = lambda s: scheduled_lr(s, 1e-5, 10, 200, 350, 50)
        self.assertAlmostEqual(lr(201), 1e-6)
        self.assertAlmostEqual(lr(210), 1e-5)
        self.assertAlmostEqual(lr(350), 1e-5)
        self.assertAlmostEqual(lr(375), 5.5e-6)
        self.assertAlmostEqual(lr(400), 1e-6)
        # A whole-run cosine is the same formula with the decay starting at step 0.
        self.assertAlmostEqual(scheduled_lr(100, 1e-5, 10, 0, 0, 200), 5.5e-6)

    def test_plateau_controller_matches_the_manual_call(self):
        self.assertEqual(replay(PlateauController()), 350)
        self.assertEqual(replay(PlateauController(patience=2)), 250)  # too eager: fires on validation noise
        controller = PlateauController(kl_ceiling=1e-2)
        self.assertEqual(replay(controller), 300)
        self.assertIn("KL", controller.reason)

    def test_plateau_state_round_trip(self):
        first = PlateauController()
        for step, val, train, kl in QWEN_RUN[:9]:
            first.update(step, val, train, kl)
        second = PlateauController()
        second.load_state_dict(first.state_dict())
        for step, val, train, kl in QWEN_RUN[9:]:
            first.update(step, val, train, kl)
            second.update(step, val, train, kl)
        self.assertEqual((first.decay_start, first.stale), (second.decay_start, second.stale))


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
        self.assertEqual(parse_reward_group('{"scores": [1, -2]}', 2), [1.0, -2.0])
        judge = SiliconFlowRewardModel("test-key")
        replies = iter(["not json", "[1, 2, 3, 3]"])
        judge._request = lambda _: next(replies)
        with patch("trainer.trainer_utils.time.sleep"):
            self.assertEqual(judge.score_group("海浪是什么？", "参考", ["a", "b", "c", "d"]), [1.0, 2.0, 3.0, 3.0])

    def test_group_judge_waits_out_rate_limits(self):
        import urllib.error
        judge = SiliconFlowRewardModel("test-key")
        replies = iter([urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None)] * 2 + ["[1, 2]"])

        def request(_):
            reply = next(replies)
            if isinstance(reply, Exception):
                raise reply
            return reply
        judge._request = request
        with patch("trainer.trainer_utils.time.sleep") as sleep:
            self.assertEqual(judge.score_group("海浪是什么？", "参考", ["a", "b"]), [1.0, 2.0])
        self.assertEqual(judge.usage["rate_limited"], 2)
        self.assertGreaterEqual(min(call.args[0] for call in sleep.call_args_list), 10)

    def test_group_judge_accepts_control_character_in_http_json(self):
        response = io.BytesIO(b'{"choices":[{"message":{"content":"[1,\n2]"}}]}')
        judge = SiliconFlowRewardModel("test-key")
        def fake_urlopen(request, timeout):
            body = json.loads(request.data)
            self.assertEqual(body["response_format"], {"type": "json_object"})
            self.assertFalse(body["enable_thinking"])
            self.assertEqual(body["model"], "Qwen/Qwen3-32B")
            self.assertIn("完全相同的答案必须同分", body["messages"][0]["content"])
            self.assertEqual(json.loads(body["messages"][1]["content"].split("\n", 1)[1])["candidates"], ["a", "b"])
            return response
        with patch("trainer.trainer_utils.urllib.request.urlopen", side_effect=fake_urlopen):
            self.assertEqual(judge.score_group("海浪是什么？", "参考", ["a", "b"]), [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
