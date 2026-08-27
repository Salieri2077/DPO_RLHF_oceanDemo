# OceanHeart 实验记录

每次训练复制一行填写；失败结果同样保留。

| Run | Git commit | Manifest SHA-256 | 阶段/数据 | 关键参数 | 最低 val loss | Ocean PPL | Generic PPL | DPO acc/margin | 结论 |
|---|---|---|---|---|---:|---:|---:|---:|---|
| baseline-pretrain | | | Pretrain / generic | E2, 8×768, S340, LR5e-4 | | | | | |
| ocean-sft-pure | | | SFT / Ocean 100% | E2, S768, LR1e-5 | | | | | |
| ocean-sft-replay | | | SFT / Ocean 90% + general 10% | E2, S768, LR1e-5 | | | | | |
| ocean-dpo | | | DPO / ocean preference | E2, S1024, LR4e-8, β0.15 | | | | | |

## 曲线诊断

- train 降、val 升：过拟合；使用 1 epoch 或回放版本。
- train/val 都在下降：尚未收敛；增加 epoch，不先改结构。
- loss 尖峰且 `train/grad_clipped > 0.05`：学习率减半。
- manifest 截断率超过 10%：序列长度改 1024，每卡 batch 1、累积 32。
- Ocean 指标改善但 Generic PPL 恶化超过 10%：采用回放版本。
- DPO accuracy/margin 未提升：先审计偏好对，再考虑增加数据。

## 本轮观察

- 数据问题：seed 42 全量处理得到 Ocean SFT 50,228 条、DPO 400 对；prompt 泄漏为 0。SFT 在 768 tokens 下抽样截断率为 45.03%，DPO 在 1024 tokens 下为 0%。
- 训练稳定性：4×2080 Ti 上微型 Pretrain、两组 SFT、DPO 均完成 20 optimizer steps，loss 有限且 SwanLab offline 正常；DPO rank 0 验证的 collective 顺序问题已由两卡续训回归覆盖。
- 泛化与遗忘：等待完整 A/B 训练后填写 Ocean/Generic PPL 变化。
- DPO 行为变化：等待完整训练后填写 accuracy 与 reward margin。
- 下一轮只改变的变量：先完成 768-token A/B 基线；随后按规则只把 SFT 改为 seq 1024、每卡 batch 1、累积 32，比较截断改善。
