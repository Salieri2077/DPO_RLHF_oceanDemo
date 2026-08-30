# OceanHeart 实验记录

每次训练复制一行填写；失败结果同样保留。

| Run | Git commit | Manifest SHA-256 | 阶段/数据 | 关键参数 | 最低 val loss | Ocean PPL | Generic PPL | DPO acc/margin | 结论 |
|---|---|---|---|---|---:|---:|---:|---:|---|
| baseline-pretrain | | | Pretrain / generic | E2, 8×768, S340, LR5e-4 | | | | | |
| ocean-sft-pure | | | SFT / Ocean 100% | E2, S768, LR1e-5 | | | | | |
| ocean-sft-replay | | | SFT / Ocean 90% + general 10% | E2, S768, LR1e-5 | | | | | |
| ocean-dpo | | | DPO / ocean preference | E2, S1024, LR4e-8, β0.15 | | | | | |
| ocean-lora | | | LoRA / Ocean 90% + general 10% | E2, r16, S768, LR1e-4 | | | | | |
| ocean-grpo-20steps | | | GRPO / Ocean + SiliconFlow judge | 20 steps, G4, S768+256, LR3e-7, β0.1 | | | | | |
| ocean-opd | | | OPD / Ocean DPO teacher | E2, S768+256, LR3e-7, clip5 | | | | | |
| general-opd | | | OPD / MiniMind SFT teacher | E2, S768+256, LR3e-7, clip5 | | | | | |
| ocean-mopd | | | MOPD / Ocean 80% + general 20% | E2, S768+256, LR3e-7, clip5 | | | | | |
| smoke-ocean-lora | 026e9b5 | c5c917ea32d09ef515d2691268acee22de7a5d89753bfd8611a62c28ae02363d | LoRA / replay | 4 GPU, 10→20 steps resume, r16 | 2.9149 | 18.45 | | | 通过；同一 SwanLab run ID 连续 |
| smoke-ocean-lora-moe | 026e9b5 | c5c917ea32d09ef515d2691268acee22de7a5d89753bfd8611a62c28ae02363d | MoE LoRA / replay | 4 GPU, 2 steps, r16, S128 | | | | | forward/backward、保存均通过 |

## 曲线诊断

- train 降、val 升：过拟合；使用 1 epoch 或回放版本。
- train/val 都在下降：尚未收敛；增加 epoch，不先改结构。
- loss 尖峰且 `train/grad_clipped > 0.05`：学习率减半。
- manifest 截断率超过 10%：序列长度改 1024，每卡 batch 1、累积 32。
- Ocean 指标改善但 Generic PPL 恶化超过 10%：采用回放版本。
- DPO accuracy/margin 未提升：先审计偏好对，再考虑增加数据。
- GRPO `group_std` 接近 0 或退化组比例高：先检查裁判区分度和候选多样性，不调大学习率。
- GRPO reward 上升但 Ocean PPL/固定回答变差：保留 SFT/DPO 权重，降低规则奖励或审计奖励投机样本。
- OPD `reverse_kl` 不降且 reward clipping 持续偏高：先检查教师权重与 Tokenizer 是否严格匹配，再考虑降低学习率或调整 clip。
- MOPD 的完成 token share 明显偏离 80/20：先按完成 token 平衡采样预算；只有确认 integration gap 来自预算失衡后才引入 Open-MOPD 调度。
- MOPD Ocean/Generic NLL 同时差于对应单域 OPD：记录 integration gap，不用增加 epoch 掩盖教师冲突或预算问题。

## 本轮观察

- 数据问题：seed 42 全量处理得到 Ocean SFT 50,228 条、DPO 400 对；prompt 泄漏为 0。SFT 在 768 tokens 下抽样截断率为 45.03%，DPO 在 1024 tokens 下为 0%。
- 训练稳定性：4×2080 Ti 上微型 Pretrain、两组 SFT、DPO 均完成 20 optimizer steps，loss 有限且 SwanLab offline 正常；DPO rank 0 验证的 collective 顺序问题已由两卡续训回归覆盖。
- 泛化与遗忘：等待完整 A/B 训练后填写 Ocean/Generic PPL 变化。
- DPO 行为变化：等待完整训练后填写 accuracy 与 reward margin。
- LoRA 烟测：Dense 4 卡先跑 10 optimizer steps，再从 checkpoint 续到 20；val loss `3.0683 → 2.9149`，PPL `21.51 → 18.45`。adapter 为 0.393M 参数（Dense 0.61%，MoE 0.20%），adapter/merged/resume 三类权重均成功保存。
- GRPO 烟测：CPU 已覆盖原生 rollout→policy backward、GRPO clip/KL、Ocean 分组 Judge 严格 JSON 与重试、微型 MoE backward。Codex 进程未继承终端中的 `SILICONFLOW_API_KEY`，因此未擅自发起付费 20-step 在线裁判训练。
- OPD/MOPD 烟测：两卡微型 Dense 完成 1→2 step checkpoint 续训，两卡 MOPD Dense/MoE 均完成 2 steps、分域指标聚合和唯一 rank-0 SwanLab offline 写入；教师 checkpoint 结构与哈希均写入续训文件。SwanLab 0.6.8 的 offline 模式不支持原 run resume，第二段 run 通过 `resumed_from_swanlab_id` 关联；cloud 模式复用原 ID。
- OPD/MOPD 评估：三个微型 checkpoint 已走通 Ocean/Generic NLL/PPL、DPO 偏好指标、确定性生成、CSV 和 80/20 integration gap 输出；微型随机权重数值只用于验证管线，不作为效果结论。
- 下一轮只改变的变量：先完成 768-token A/B 基线；随后按规则只把 SFT 改为 seq 1024、每卡 batch 1、累积 32，比较截断改善。
