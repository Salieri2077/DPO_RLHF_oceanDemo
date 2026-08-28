# Dense vs MoE Pretrain Baseline

评测日期：2026-08-28；Git 基线：`db46955`；seed 42；FP16；max sequence length 340。主指标是按 continuation token 数归一化的 candidate accuracy。

| Domain | N | Dense | MoE | MoE - Dense | Paired bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| General English | 500 | 27.6% | 28.2% | +0.6 pp | [-1.8, +3.0] pp |
| General Chinese | 500 | 52.6% | 54.2% | +1.6 pp | [-1.8, +4.8] pp |
| Science / Reasoning | 500 | 30.4% | 31.0% | +0.6 pp | [-2.6, +3.8] pp |
| Ocean | 602 | 23.6% | 24.8% | +1.2 pp | [-2.0, +4.2] pp |
| Overall (micro) | 2,102 | 33.1% | 34.1% | +1.0 pp | [-0.5, +2.5] pp |

结论：MoE 在四个领域的点估计都略高，但所有 95% 区间都跨过 0，当前证据只能说明两者接近，不能声称 MoE 胜出。Ocean 的 +1.2 pp 也不显著，尚无“MoE 尤其擅长海洋”的证据。

正确候选 PPL 方面 Dense 为 28.08、MoE 为 29.90（MoE 高 6.5%），说明 MoE 的微弱命中率增益没有转化为更好的正确答案似然。Dense 有 63.91M 总/激活参数；MoE 有 198.42M 总参数、每 token 约 63.94M 激活参数，因此这里只控制激活计算规模，并非总参数量相同。

各 benchmark 明细和逐题分数位于本机 `Pretrain Evaluation/results/`。HellaSwag、ARC、Ocean 两套题的结果接近四选一随机水平，XCOPA 接近二选一随机水平，符合 64M 级 Base Model 能力有限的预期；不要把小幅差值包装成能力突破。

泄漏审计对 2,102 个规范化 prompt 和实际预训练文件做了 SHA-256 精确匹配，发现 0 条重合。该结果不能排除子串、改写、语义重复或原始上游数据污染。
