# GRPO Evaluation

固定使用 `ocean_grpo_val.jsonl` 的前 50 道海洋问题，比较：

- MoE SFT：`out/ocean_sft_replay_768_moe.pth`
- MoE GRPO：`out/ocean_grpo_eval50_768_moe.pth`

两个模型依次加载并使用 greedy 解码，避免采样噪声和同时占用显存。SiliconFlow judge 在参考答案辅助下同时给两个回答打 `[-3, 3]` 分；候选顺序逐题交替，降低位置偏差。主结果是逐题 `GRPO - SFT` 分差、胜率和 2,000 次配对 bootstrap 95% 区间。

```bash
cd /home/anhuang/OceanHeart
export SILICONFLOW_API_KEY="你的密钥"
/home/anhuang/.conda/envs/minimind/bin/python "GRPO Evaluation/evaluator.py"
```

烟测可加 `--samples 1 --bootstrap-repeats 100`。输出位于 `GRPO Evaluation/results/moe/`：

- `generations_and_scores.jsonl`：问题、参考答案、两份回答与逐题评分；
- `comparison.csv`：便于表格分析的扁平结果；
- `summary.json`：协议、哈希、平均分、胜率与置信区间；
- `comparison.md`：简要结论。

历史50题结果使用 `Qwen/Qwen3-8B`。当前默认升级为 `Qwen/Qwen3-32B` 并使用明确评分锚点；新旧分数不可直接混合比较。与训练共享裁判仍不能替代独立裁判或人工盲评。

训练前运行真实 API 验收（24次请求）：

```bash
/home/anhuang/.conda/envs/minimind/bin/python "GRPO Evaluation/validate_judge.py"
```

验收包含4组历史重复答案、相同正确答案、4候选好坏排序，每组正反序各重复两次。相同答案必须同分，近似低质答案均不高于-2且分差不超过0.5，换序分差不超过0.5，好答案至少2分。原始评分写入 `results/judge32b_gate.json`；任何检查失败时返回非零退出码，不启动训练。通过仅代表这些用例通过，不代表所有领域评分都可靠。

运行单元测试：

```bash
cd "GRPO Evaluation"
/home/anhuang/.conda/envs/minimind/bin/python -m unittest -v
```
