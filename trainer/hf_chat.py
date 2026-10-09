"""Chat formatting and LoRA loading shared by the Hugging Face (Qwen) Ocean SFT, evaluation and RL scripts.

Plain ChatML without <think> blocks: the base model learns one answer format, and generation prompts end
with "<|im_start|>assistant\\n" exactly as in training. Loss covers assistant content plus its end token.

Assistant replies end with <|endoftext|>, not <|im_end|>. In Qwen3-1.7B-Base the <|im_start|>/<|im_end|>
embeddings were never trained: they equal ~4.8K other untrained rows (cosine 1.0). With tied input/output
embeddings their logits are identical, so a model trained to emit <|im_end|> picks one of those twins at
random (e.g. "𫟦") and never stops. <|endoftext|> is the trained pretraining document separator.
"""
from pathlib import Path

import torch

CHATML = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}"
          "{{ '<|endoftext|>' if m['role'] == 'assistant' else '<|im_end|>' }}\n{% endfor %}"
          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
END = "<|im_end|>"
ANSWER_END = "<|endoftext|>"


def turn_end(role):
    return ANSWER_END if role == "assistant" else END


def encode_conversation(tokenizer, conversations, max_len):
    """Token ids, labels (-100 outside assistant replies) and the labeled character count."""
    ids, labels, chars = [], [], 0
    for message in conversations:
        head = tokenizer(f"<|im_start|>{message['role']}\n", add_special_tokens=False).input_ids
        if message["role"] == "assistant":
            body = tokenizer(message["content"] + ANSWER_END, add_special_tokens=False).input_ids
            ids += head + body
            labels += [-100] * len(head) + body
            chars += len(message["content"])
        else:
            body = tokenizer(message["content"] + END, add_special_tokens=False).input_ids
            ids += head + body
            labels += [-100] * (len(head) + len(body))
        newline = tokenizer("\n", add_special_tokens=False).input_ids
        ids += newline
        labels += [-100] * len(newline)
    return ids[:max_len], labels[:max_len], chars


def generation_prompt(conversations):
    """Prompt text for every message before the final assistant reply."""
    messages = conversations[:-1] if conversations[-1]["role"] == "assistant" else conversations
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}{turn_end(m['role'])}\n" for m in messages) + "<|im_start|>assistant\n"


def load_tokenizer(model_path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.chat_template = CHATML
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(model_path, adapter=None, device="cuda", dtype=torch.float16):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, Path(adapter))
    return model.eval()
