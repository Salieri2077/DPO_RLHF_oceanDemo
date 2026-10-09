# Qwen3-1.7B-Base 海洋 SFT（2026-10-08）

目的：把 64M 自研模型的 Ocean SFT 换成成熟基座重做一遍，判断"基座太小"是否是后训练效果差的主因，并作为后续 RL / Agent 链路的新起点。64M 链路保留为小基座对照。

## 设置

| 项 | 取值 |
|---|---|
| 基座 | `Qwen/Qwen3-1.7B-Base`（rev `ea980cb`，Apache-2.0），本地 `/data/anhuang/oceanheart_models/qwen3-1.7b-base` |
| 数据 | 与 64M 相同：`ocean_sft_replay_train.jsonl`（40,152 Ocean + 10% 通用回放 = 44,614 条），验证 `ocean_sft_val.jsonl` 前 512 条 |
| 格式 | ChatML（无 think），系统/用户轮以 `<\|im_end\|>` 结尾，**助手轮以 `<\|endoftext\|>` 结尾**（原因见下） |
| 长度 | 1024，Qwen 分词下 p50 447、最大 1010，**0% 截断**（64M 在 768 下截断 45%） |
| 训练 | LoRA r64 / α128，全部线性层，6,973 万可训练参数（4%）；fp16 基座 + fp32 LoRA（2080 Ti 无 bf16）；仅助手位置算 loss，分块计算词表 |
| 超参 | 2 epochs，4 卡 × 16 = 64 条/步，1,396 步，LR 2e-4，30 步 warmup，余弦到 10% |
| 资源 | 每步约 3.6 s，峰值显存 6.8 GiB，总计 1.58 h |
| 代码 | `trainer/train_sft_hf.py`、`trainer/hf_chat.py`；评估 `SFT Evaluation/compare_backbones.py` |
| SwanLab | OceanHeart-SFT / `qwen3-1.7b-base-ocean-sft-r64-eot-20261008-2006` |

fp16 数值检查：基座在 16 条验证样本上 fp32 loss 1.72765，fp16 1.72772。

## 踩坑：Base 模型的 `<|im_end|>` 未训练

第一轮（`...-r64-20261008-1633`）按标准 ChatML 用 `<|im_end|>` 作为回答结束符，SFT 后 100 题 **0% 正常结束**，在结尾处输出 `𫟦`、` bakeka` 等生僻 token 后继续生成。

原因：Qwen3-1.7B-Base 中 `<|im_start|>`、`<|im_end|>` 的嵌入从未训练，范数 0.375（中位数 1.62），与约 4,800 个未训练行**完全相同**（余弦 1.0）。模型输入输出嵌入共享（`tie_word_embeddings`），这些行的 logit 恒相等，结束符最多只能分到约 1/4,800 的概率（实测 0.0003）。LoRA 不改嵌入，无法修复。

修复：助手回答改以 `<|endoftext|>`（预训练文档分隔符，范数 1.88，已训练）结尾。第 100 步抽查 5 题结尾处 P(`<|endoftext|>`) 为 0.89～0.996，最终生成 100% 正常结束。第一轮结果归档在 `results/backbones/imend_bug_run1/`。

## 结果

Ocean 测试集前 500 题算 NLL、前 100 题 greedy 生成；通用集为 `generic_sft_eval.jsonl` 前 500 条。不同分词器的 token PPL 不可比，跨模型比较用每字符 NLL（nats/char）。

| 指标 | 64M SFT | Qwen3-1.7B-Base（零样本） | Qwen3-1.7B SFT |
|---|---:|---:|---:|
| Ocean 测试 nats/char ↓ | 0.791 | 0.375 | **0.286** |
| 通用 nats/char ↓ | 1.169 | 0.492 | **0.450** |
| 正常结束率 | 54% | 98% | **100%** |
| 回答语言与问题一致 | 99% | 93% | **100%** |
| 6-gram 重复率 > 0.3 的比例（粗指标，见下） | 100% | **19%** | 35% |
| DeepSeek-V3 裁判均分（−3～3） | −2.90 | 1.605 [1.32, 1.87] | 1.610 [1.31, 1.90] |

裁判协议：每题三个候选同一请求打分，种子随机顺序与倒序各一次取平均；19/300 个候选两次相差 > 1 分。

配对比较（95% bootstrap 区间）：
- SFT − Base：+0.005 [−0.32, +0.34]，胜 43 / 负 32 / 平 25，**无显著差异**。
- SFT − 64M：+4.51 [+4.22, +4.79]，100 胜 0 负。

分语言：中文 49 题 SFT 1.67 vs Base 1.45；英文 51 题 SFT 1.55 vs Base 1.75。

## 解读

1. **基座是 64M 链路的主因，得到确认。** 同样的数据和 SFT 流程，1.7B 从 −2.9 到 +1.6；未微调的 Qwen Base 也远好于 64M SFT。
2. **SFT 学到了领域分布和格式，但裁判分没有超过 Base。** Ocean nats/char 降 24%，通用能力未受损，回答格式、语言、结束行为都更规范；但均分与 Base 持平。
3. **持平来自两种变化互相抵消。** SFT 的 3 分题从 20 增到 30，−1 分及以下从 13 增到 18。OceanInstruct 的参考答案风格是"具体、分点、带数值"，模型学会了这种自信的写法，但在不掌握的知识上编造细节，例如把反气旋说成气旋、"夜晚无光照时进行光合作用"。这与"SFT 塞入新知识会增加幻觉"的已有观察一致。
4. **"重复率 > 0.3"会高估英文复读。** 英文参考答案自身的 6-gram 重复率中位数就有 0.18（同一术语反复出现）。真正的循环很少：SFT 100% 正常结束，同一行原样出现 ≥3 次的只有 3 题（Base 也是 3 题）。按同语言参考答案的 95% 分位数校准后：中文 SFT 超标 23/49 题，少于 Base 的 35/49；英文 SFT 超标 34/51 题，多于 Base 的 17/51。SFT 的英文回答确实比参考答案啰嗦（多为同一短语反复出现），与英文题裁判分低于 Base 一致。

## best-of-8 检查（GRPO 前）

脚本 `GRPO Evaluation/best_of_n_qwen.py`，结果在 `GRPO Evaluation/results/qwen_best_of_8/`。题目：GRPO 验证集前 20 题（SFT 未见）和 GRPO 训练集前 10 题（SFT 训练见过）。每题生成 1 个 greedy 回答和 8 个 T=0.8 采样回答，9 个一起由 DeepSeek-V3 打分，正反序各一次取平均。

| | 验证 20 题 | 训练 10 题 |
|---|---:|---:|
| 采样均分 | 1.67 | 1.48 |
| best-of-8 | 2.43 | 2.15 |
| worst-of-8 | 0.63 | 0.65 |
| greedy | 2.28 | 1.90 |
| 组内标准差均值 | 0.62 | 0.56 |
| 无信号组（std < 0.25） | 15% | 30% |
| 组内最高最低差 ≥ 2 | 45% | 40% |
| 最好样本也 ≤ 0 分 | 0% | 10% |

- 与 64M 相反，**绝大多数组有好有坏**，GRPO 有可学的信号。
- **采样明显差于 greedy**：验证集 1.67 对 2.28；8 个样本中最好的超过 greedy 的只有 11/30 组。T=0.8 采样会带出编造，例如 val-4 greedy 得 3 分，样本最低 −1 分。
- 因此 GRPO 的合理预期是：采样回答均分向 best-of-8 靠拢，最差样本变少，即编造减少；greedy 的提升空间小，验证集上 best-of-8 只比 greedy 高 0.15。
- 样本得分与重复率（r = −0.01）、长度（r = 0.05）都几乎不相关。奖励中不需要重复惩罚，裁判也没有偏好长回答。
- 少数难题所有样本都低（val-3、val-19、train-1），属于知识缺口，RL 无法弥补。
- 裁判噪声：270 个候选中有 23 个正反序相差 > 1 分。
- 样本量小（30 题），以上只作开训判断，不作为最终结论。

## 复现

```bash
cd /home/anhuang/OceanHeart/trainer
torchrun --standalone --nproc_per_node=4 train_sft_hf.py \
  --model_path /data/anhuang/oceanheart_models/qwen3-1.7b-base \
  --save_adapter qwen3_1.7b_ocean_sft_r64_eot --run_name <tag> \
  --epochs 2 --eval_interval 100 --val_samples 512 \
  --use_swanlab --swanlab_mode cloud --swanlab_project OceanHeart-SFT

cd /home/anhuang/OceanHeart
python "SFT Evaluation/compare_backbones.py" --stage generate --models qwen3_1.7b_sft \
  --adapter out/hf/qwen3_1.7b_ocean_sft_r64_eot_best   # 另两个模型同理，可分卡并行
SILICONFLOW_API_KEY=$(cat ~/.siliconflow_key) python "SFT Evaluation/compare_backbones.py" --stage judge
```
