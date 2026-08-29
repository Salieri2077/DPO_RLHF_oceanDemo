# SFT Evaluation

这里固定比较同一种架构的三个 checkpoint：

| 名称 | 权重 | 作用 |
|---|---|---|
| Pretrain | `pretrain_768[ _moe].pth` | SFT 前基线，用来判断指令微调带来的净变化 |
| Pure | `ocean_sft_pure_768[ _moe].pth` | 仅 OceanInstruct，观察海洋领域拟合上限 |
| Replay | `ocean_sft_replay_768[ _moe].pth` | 90% OceanInstruct + 10% 通用 SFT 回放，观察是否缓解遗忘 |

方括号中的 `_moe` 表示选择 MoE 架构时自动使用的后缀，不是文件名中的空格。

## 评估内容

| 维度 | 数据 | 指标 | 回答的问题 |
|---|---|---|---|
| 海洋回答拟合 | `data/processed/ocean_sft_test.jsonl` | assistant-token NLL / PPL | 哪个模型更能复现未见过的海洋参考回答 |
| 通用能力保持 | `data/processed/generic_sft_eval.jsonl` | assistant-token NLL / PPL | Pure 是否发生更明显的通用能力遗忘 |
| 多领域选择题 | `/data/anhuang/oceanheart_pretrain_eval` 的 5 个现有 benchmark | chat-template candidate accuracy、长度归一化 accuracy（主指标） | 海洋增强是否以牺牲常识、中文和科学推理为代价 |
| 开放式回答 | 海洋测试集固定前 50 题 | greedy 生成、吞吐、人工查看 | loss 接近时，实际回答风格和内容有何差异 |
| 可选模型裁判 | 同一批开放式回答 | Pure/Replay 胜负与平均分 | 参考答案辅助下的事实性、相关性和清晰度 |

NLL/PPL 是 teacher-forced 指标，低不等于自由生成一定更正确，因此同时保留选择题和确定性生成。选择题会套用 OceanHeart system prompt 和 tokenizer 的 chat template；这与 `Pretrain Evaluation` 面向 Base Model、故意不套 chat template 的协议不同。

Pure 与 Replay 的每样本 NLL差、选择题正确率差都会进行 2,000 次配对 bootstrap，并输出 95% 区间。区间跨过 0 时只写 `Inconclusive`，不把很小的随机差异解释成胜负。

## 运行

Dense：

```bash
cd "/home/anhuang/OceanHeart"
/home/anhuang/.conda/envs/minimind/bin/python \
  "SFT Evaluation/evaluator.py" --architecture dense
```

MoE：

```bash
/home/anhuang/.conda/envs/minimind/bin/python \
  "SFT Evaluation/evaluator.py" --architecture moe
```

先验证流程可加 `--max-samples 5 --generation-samples 2`。正式对比时不要给 Pure 和 Replay 使用不同参数。默认复用现有处理后数据和 `/data/anhuang/oceanheart_pretrain_eval`，不会下载或改写数据。

可选 SiliconFlow 裁判只比较 Pure 与 Replay，候选顺序逐题交替以减轻位置偏差：

```bash
export SILICONFLOW_API_KEY="你的密钥"
/home/anhuang/.conda/envs/minimind/bin/python \
  "SFT Evaluation/evaluator.py" --architecture dense --judge siliconflow
```

无密钥时核心离线评估不受影响。API 调用会产生费用，建议先用较小的 `--generation-samples` 验证。

## 输出与解读

结果默认写入被 Git 忽略的 `SFT Evaluation/results/dense|moe/`：

- 每个 checkpoint 的 `summary.json`、逐样本 NLL 和选择题预测；
- `generations.jsonl`：三个 checkpoint 对相同问题的确定性回答；
- `comparison.json/csv/md`：Pure vs Replay 的配对比较；
- 使用裁判时额外生成 `judge_summary.json`。

优先检查：Replay 的海洋指标是否接近 Pure，同时通用 PPL 更低、通用 benchmark 不退化。如果 Pure 海洋更好但通用显著退化，Replay 是更稳妥的后续 DPO 起点；如果差异区间跨 0，应结论为当前证据不足。

当前两次训练是“相同 epoch”，不是严格“相同 token 预算”：Pure 有 40,152 条训练样本，Replay 有 44,614 条，因此 Replay 约多 11% optimizer steps。需要严格归因时，再补一轮固定 `max_steps` 的对照。

运行最小测试：

```bash
cd "SFT Evaluation" && /home/anhuang/.conda/envs/minimind/bin/python -m unittest -v
```
