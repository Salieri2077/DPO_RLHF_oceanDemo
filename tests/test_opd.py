import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from scripts.prepare_opd_data import load_jsonl, prepare, prompt_key
from trainer.rollout_engine import completion_log_probs
from trainer.train_opd import load_teacher, opd_objective, routed_teacher_log_probs
from trainer.trainer_utils import swanlab_run_id


def conversation(question, answer):
    return {
        "conversations": [
            {"role": "system", "content": "你是 OceanHeart。"},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


class OPDTest(unittest.TestCase):
    def test_swanlab_ids_are_unique_and_resume_is_stable(self):
        first, resume_first = swanlab_run_id(None)
        second, _ = swanlab_run_id(None)
        self.assertEqual(len(first), 21)
        self.assertNotEqual(first, second)
        self.assertEqual(resume_first, "allow")
        self.assertEqual(swanlab_run_id({"swanlab_id": first}), (first, "must"))
        self.assertEqual(swanlab_run_id(None, "offline"), (None, None))

    def test_reward_clipping_stop_gradient_and_backward(self):
        new = torch.zeros(1, 2, requires_grad=True)
        old = torch.zeros_like(new)
        teacher = torch.tensor([[10.0, 2.0]], requires_grad=True)
        metrics = opd_objective(new, old, teacher, torch.ones_like(new), reward_clip=5)
        self.assertAlmostEqual(metrics["opd_reward"].item(), 3.5)
        self.assertAlmostEqual(metrics["reward_clip_fraction"].item(), 0.5)
        metrics["policy_loss"].backward()
        self.assertIsNotNone(new.grad)
        self.assertIsNone(teacher.grad)

    def test_teacher_routing_and_tiny_dense_moe_backward(self):
        for use_moe in (False, True):
            config = MiniMindConfig(hidden_size=64, num_hidden_layers=1, use_moe=use_moe)
            student = MiniMindForCausalLM(config)
            teachers = {
                "ocean": MiniMindForCausalLM(config).eval().requires_grad_(False),
                "general": MiniMindForCausalLM(config).eval().requires_grad_(False),
            }
            output_ids = torch.randint(0, config.vocab_size, (2, 6))
            completion_ids = output_ids[:, -2:]
            attention = torch.ones_like(output_ids)
            old = completion_log_probs(student, output_ids, completion_ids, attention).detach()
            teacher = routed_teacher_log_probs(
                teachers, ["ocean", "general"], output_ids, completion_ids, attention
            )
            output = student(output_ids, attention_mask=attention, logits_to_keep=3)
            new = torch.log_softmax(output.logits[:, :-1], -1).gather(
                2, completion_ids.unsqueeze(-1)
            ).squeeze(-1)
            metrics = opd_objective(new, old, teacher, torch.ones_like(new, dtype=torch.bool))
            loss = metrics["policy_loss"] + output.aux_loss
            self.assertTrue(math.isfinite(loss.item()))
            loss.backward()

    def test_teacher_checkpoint_validation(self):
        config = MiniMindConfig(hidden_size=64, num_hidden_layers=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.pth"
            torch.save(MiniMindForCausalLM(config).state_dict(), path)
            teacher = load_teacher(config, path, "cpu", torch.float16)
            self.assertFalse(next(teacher.parameters()).requires_grad)
            bad = Path(directory) / "bad.pth"
            torch.save({}, bad)
            with self.assertRaises(ValueError):
                load_teacher(config, bad, "cpu", torch.float16)
            with self.assertRaises(FileNotFoundError):
                load_teacher(config, Path(directory) / "missing.pth", "cpu", torch.float16)

    def test_stable_two_domain_data_and_zero_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write(name, items):
                with (root / name).open("w", encoding="utf-8") as handle:
                    for item in items:
                        handle.write(json.dumps(item, ensure_ascii=False) + "\n")

            ocean_train = [conversation(f"海洋问题{i}", f"海洋答案{i}") for i in range(4)]
            ocean_val = [conversation(f"海洋验证{i}", f"海洋答案{i}") for i in range(2)]
            general_train = [conversation(f"通用问题{i}", f"通用答案{i}") for i in range(3)]
            general_val = [conversation(f"通用验证{i}", f"通用答案{i}") for i in range(2)]
            write("ocean_grpo_train.jsonl", ocean_train)
            write("ocean_grpo_val.jsonl", ocean_val)
            write("ocean_sft_train.jsonl", ocean_train)
            write("ocean_sft_replay_train.jsonl", ocean_train + general_train)
            write("generic_sft_eval.jsonl", general_val)

            first = prepare(root, seed=42, ocean_train_count=2, general_train_count=1, val_count=1)
            second = prepare(root, seed=42, ocean_train_count=2, general_train_count=1, val_count=1)
            self.assertEqual(first["outputs"], second["outputs"])
            train = load_jsonl(root / "ocean_mopd_train.jsonl")
            val = load_jsonl(root / "ocean_mopd_val.jsonl")
            self.assertEqual([item["domain"] for item in train].count("ocean"), 2)
            self.assertEqual([item["domain"] for item in train].count("general"), 1)
            self.assertFalse({prompt_key(item) for item in train} & {prompt_key(item) for item in val})


if __name__ == "__main__":
    unittest.main()
