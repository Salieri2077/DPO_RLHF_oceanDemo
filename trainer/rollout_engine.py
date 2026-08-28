"""Native PyTorch rollout used by OceanHeart GRPO."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel


@dataclass
class RolloutResult:
    output_ids: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    old_log_probs: torch.Tensor
    completions: list[str]


def completion_log_probs(model, output_ids, completion_ids, attention_mask=None):
    logits = model(
        output_ids, attention_mask=attention_mask, logits_to_keep=completion_ids.size(1) + 1
    ).logits[:, :-1]
    return F.log_softmax(logits, dim=-1).gather(2, completion_ids.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def rollout(model, tokenizer, prompt_ids, attention_mask, num_generations, max_new_tokens, temperature=0.8):
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    was_training = raw.training
    raw.eval()
    repeated_ids = prompt_ids.repeat_interleave(num_generations, 0)
    repeated_mask = attention_mask.repeat_interleave(num_generations, 0)
    output_ids = raw.generate(
        input_ids=repeated_ids,
        attention_mask=repeated_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    ).clone()
    prompt_length = prompt_ids.size(1)
    completion_ids = output_ids[:, prompt_length:]
    completion_mask = completion_ids.ne(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None and completion_ids.numel():
        eos = completion_ids.eq(tokenizer.eos_token_id) & completion_mask
        end = completion_ids.new_full((completion_ids.size(0),), completion_ids.size(1) - 1)
        end[eos.any(1)] = eos.int().argmax(1)[eos.any(1)]
        completion_mask &= torch.arange(completion_ids.size(1), device=completion_ids.device).unsqueeze(0) <= end.unsqueeze(1)
    full_mask = output_ids.ne(tokenizer.pad_token_id).long()
    old_log_probs = completion_log_probs(raw, output_ids, completion_ids, full_mask)
    completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    if was_training:
        raw.train()
    return RolloutResult(output_ids, completion_ids, completion_mask, old_log_probs, completions)
