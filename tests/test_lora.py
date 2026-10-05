import tempfile
import unittest
from pathlib import Path

import torch

from model.model_lora import apply_lora, load_lora, save_lora, save_merged_lora
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


class LoRATest(unittest.TestCase):
    def test_freeze_save_load_and_merge(self):
        config = MiniMindConfig(hidden_size=64, num_hidden_layers=1)
        model = MiniMindForCausalLM(config).eval()
        base_state = {name: value.clone() for name, value in model.state_dict().items()}
        self.assertEqual(apply_lora(model, rank=4), 2)
        for name, parameter in model.named_parameters():
            parameter.requires_grad = ".lora." in name
        self.assertTrue(all(parameter.requires_grad == (".lora." in name) for name, parameter in model.named_parameters()))
        for name, parameter in model.named_parameters():
            if name.endswith("lora.B.weight"):
                parameter.data.normal_(std=0.01)
        inputs = torch.randint(0, config.vocab_size, (1, 8))
        expected = model(inputs).logits
        expected.mean().backward()
        self.assertTrue(any(parameter.grad is not None for name, parameter in model.named_parameters() if name.endswith("lora.B.weight")))
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapter.pth"
            merged = Path(directory) / "merged.pth"
            save_lora(model, adapter)
            reloaded = MiniMindForCausalLM(config)
            reloaded.load_state_dict(base_state)
            apply_lora(reloaded, rank=4)
            load_lora(reloaded, adapter)
            save_merged_lora(model, merged)
            plain = MiniMindForCausalLM(config).eval()
            plain.load_state_dict(torch.load(merged, map_location="cpu"))
        self.assertTrue(torch.allclose(expected, reloaded.eval()(inputs).logits, atol=2e-3, rtol=2e-3))
        self.assertTrue(torch.allclose(expected, plain(inputs).logits, atol=2e-3, rtol=2e-3))


if __name__ == "__main__":
    unittest.main()
