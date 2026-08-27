import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.lm_dataset import DPODataset, PretrainDataset, SFTDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from scripts.prepare_ocean_data import convert_dpo, convert_sft, digest, ocean_split, prepare_ocean_sft
from trainer.train_dpo import dpo_metrics


class DataConversionTest(unittest.TestCase):
    def test_sft_conversion_and_stable_split(self):
        raw = {"conversations": [{"from": "human", "value": "什么是潮汐？"}, {"from": "gpt", "value": "潮汐是海面周期变化。"}]}
        item = convert_sft(raw)
        self.assertEqual([message["role"] for message in item["conversations"]], ["system", "user", "assistant"])
        self.assertEqual(ocean_split("什么是潮汐？", 42), ocean_split(" 什么是潮汐？\n", 42))
        self.assertEqual(digest(item), digest(item))

    def test_invalid_and_corrupt_sft_are_rejected(self):
        self.assertIsNone(convert_sft({"conversations": [{"from": "human", "value": "坏\ufffd数据"}]}))
        self.assertIsNone(convert_sft({"conversations": [{"from": "gpt", "value": "角色顺序错误"}]}))

    def test_dpo_becomes_two_complete_dialogues(self):
        item = convert_dpo({
            "conversations": [{"from": "human", "value": "解释海啸。"}],
            "chosen": {"from": "gpt", "value": "海啸通常由海底扰动引发。"},
            "rejected": {"from": "gpt", "value": "海啸就是潮汐。"},
        })
        self.assertEqual([message["role"] for message in item["chosen"]], ["system", "user", "assistant"])
        self.assertNotEqual(item["chosen"][-1]["content"], item["rejected"][-1]["content"])
        self.assertIsNone(convert_dpo({
            "context": [{"role": "assistant", "content": "顺序错误"}],
            "chosen": {"role": "assistant", "content": "A"},
            "rejected": {"role": "assistant", "content": "B"},
        }))
        self.assertIsNone(convert_dpo({
            "context": [{"role": "user", "content": "问题"}],
            "chosen": {"role": "assistant", "content": "相同"},
            "rejected": {"role": "assistant", "content": " 相同 "},
        }))

    def test_ocean_dedup_and_prompt_isolation(self):
        first = {"conversations": [{"from": "human", "value": "什么是潮汐？"}, {"from": "gpt", "value": "答案一"}]}
        second = {"conversations": [{"from": "human", "value": " 什么是潮汐？\n"}, {"from": "gpt", "value": "答案二"}]}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "ocean.json"
            source.write_text(json.dumps([first, first, second], ensure_ascii=False), encoding="utf-8")
            splits, stats = prepare_ocean_sft(source, 42)
        self.assertEqual(stats["duplicate"], 1)
        populated = [name for name, items in splits.items() if items]
        self.assertEqual(len(populated), 1, "normalized duplicate prompts must stay in one split")


class TrainingSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
        cls.model = MiniMindForCausalLM(MiniMindConfig(hidden_size=64, num_hidden_layers=1))

    def write_jsonl(self, directory, name, items):
        path = Path(directory) / name
        with path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        return path

    def test_pretrain_sft_and_dpo_losses_are_finite(self):
        with tempfile.TemporaryDirectory() as directory:
            pretrain_path = self.write_jsonl(directory, "pretrain.jsonl", [{"text": "海洋覆盖了地球表面的大部分区域。"}])
            conversation = {"conversations": [{"role": "user", "content": "什么是潮汐？"}, {"role": "assistant", "content": "潮汐是海面周期性涨落。"}]}
            sft_path = self.write_jsonl(directory, "sft.jsonl", [conversation])
            dpo_path = self.write_jsonl(directory, "dpo.jsonl", [{
                "chosen": conversation["conversations"],
                "rejected": [{"role": "user", "content": "什么是潮汐？"}, {"role": "assistant", "content": "潮汐是一种鱼。"}],
            }])

            for dataset in (
                PretrainDataset(pretrain_path, self.tokenizer, max_length=32),
                SFTDataset(sft_path, self.tokenizer, max_length=64, deterministic=True),
            ):
                input_ids, labels = dataset[0]
                output = self.model(input_ids.unsqueeze(0), labels=labels.unsqueeze(0))
                self.assertTrue(math.isfinite(output.loss.item()))
                output.loss.backward()
                self.model.zero_grad(set_to_none=True)

            deterministic_sft = SFTDataset(sft_path, self.tokenizer, max_length=64, deterministic=True)
            self.assertTrue(torch.equal(deterministic_sft[0][1], deterministic_sft[0][1]))
            batch = DPODataset(dpo_path, self.tokenizer, max_length=64, deterministic=True)[0]
            x = torch.stack([batch["x_chosen"], batch["x_rejected"]])
            y = torch.stack([batch["y_chosen"], batch["y_rejected"]])
            mask = torch.stack([batch["mask_chosen"], batch["mask_rejected"]])
            logits = self.model(x).logits
            token_log_probs = torch.gather(torch.log_softmax(logits, -1), 2, y.unsqueeze(2)).squeeze(2)
            policy = (token_log_probs * mask).sum(1)
            metrics = dpo_metrics(policy.detach(), policy, beta=0.15)
            self.assertTrue(math.isfinite(metrics["dpo_loss"].item()))
            metrics["dpo_loss"].backward()


if __name__ == "__main__":
    unittest.main()
