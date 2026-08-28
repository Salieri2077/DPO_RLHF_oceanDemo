"""Minimal LoRA adapters for MiniMind attention projections."""

import os

import torch
from torch import nn


class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, inputs):
        return self.B(self.A(inputs))


def apply_lora(model, rank=16):
    """Attach adapters only to MiniMind q_proj/o_proj layers."""
    if rank < 1:
        raise ValueError("LoRA rank must be positive")
    targets = 0
    device = next(model.parameters()).device
    for name, module in model.named_modules():
        if name.endswith(("q_proj", "o_proj")) and isinstance(module, nn.Linear):
            module.lora = LoRA(module.in_features, module.out_features, rank).to(device)
            original_forward = module.forward

            def forward_with_lora(inputs, base=original_forward, adapter=module.lora):
                return base(inputs) + adapter(inputs)

            module.forward = forward_with_lora
            targets += 1
    if not targets:
        raise ValueError("No q_proj/o_proj layers found for LoRA")
    return targets


def lora_state_dict(model):
    raw = getattr(getattr(model, "module", model), "_orig_mod", getattr(model, "module", model))
    return {
        name: value.detach().half().cpu()
        for name, value in raw.state_dict().items()
        if ".lora." in name
    }


def save_lora(model, path):
    temp = str(path) + ".tmp"
    torch.save(lora_state_dict(model), temp)
    os.replace(temp, path)


def load_lora(model, path):
    state = torch.load(path, map_location=next(model.parameters()).device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(".lora." in name for name in missing):
        raise RuntimeError(f"Invalid LoRA checkpoint: missing={missing}, unexpected={unexpected}")


def merged_state_dict(model):
    raw = getattr(getattr(model, "module", model), "_orig_mod", getattr(model, "module", model))
    state = {name: value.detach().half().cpu() for name, value in raw.state_dict().items() if ".lora." not in name}
    for name, module in raw.named_modules():
        if hasattr(module, "lora"):
            state[f"{name}.weight"] += (module.lora.B.weight @ module.lora.A.weight).detach().half().cpu()
    return state


def save_merged_lora(model, path):
    temp = str(path) + ".tmp"
    torch.save(merged_state_dict(model), temp)
    os.replace(temp, path)
