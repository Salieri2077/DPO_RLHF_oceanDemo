"""Memory-efficient SDPA for Qwen3 (grouped KV heads) on GPUs without FlashAttention (2080 Ti, sm75).

With no attention mask, transformers asks SDPA for enable_gqa, which in torch 2.6 only flash (absent on sm75) and the
math kernel support, so the full L x L attention is materialised: generation runs out of memory at about 8K tokens.
Expanding the KV heads first lets SDPA use the memory-efficient kernel (16K-token prompt: 5.8 GiB peak; greedy tokens
unchanged on a spot check). Call once per process before running the model.
"""


def memory_efficient_attention():
    import transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa = lambda *args, **kwargs: False
