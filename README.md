# OceanHeart

OceanHeart 是一个基于 [MiniMind](https://github.com/jingyaogong/minimind) 训练骨架的 63.9M 参数海洋领域文本语言模型实验项目。目标不是隐藏上游实现，而是完整复现并分析：通用预训练 → 海洋 SFT/LoRA → 海洋 DPO/GRPO。

旧版 Qwen2.5-7B + LLaMA-Factory LoRA 工程保存在 Git 标签 `legacy-llamafactory-v1`；当前版本只保留可复现的数据、训练、评估主链。

## 环境与数据

使用已经验证的本地环境：

```bash
conda activate /home/anhuang/.conda/envs/minimind
cd /home/anhuang/OceanHeart
python scripts/prepare_ocean_data.py
python scripts/prepare_ocean_grpo.py
```

数据处理会读取：

- `data/raw/ocean_instruct/OceanInstruct-v0.2.json`
- `data/raw/ocean_dpo_data.json`
- `/home/anhuang/minimind/dataset/pretrain_t2t_mini.jsonl`
- `/home/anhuang/minimind/dataset/sft_t2t_mini.jsonl`

生成文件位于 `data/processed/`，不会提交到 Git。`manifest.json` 记录数据来源、许可、稳定切分、过滤数量、token 长度和截断率。Ocean 数据按规范化问题哈希做 80/10/10 切分，DPO 与 SFT 的相同问题始终进入同一集合。

## Dense 与 MoE 权重规则

`--from_weight` 传的是权重名前缀，不是完整路径。训练器会在 `--save_dir`（默认 `../out`）中按模型结构自动补齐文件名：

| 阶段 | 参数 | Dense 实际文件 | `--use_moe 1` 实际文件 |
|---|---|---|---|
| Pretrain 输出 | `--save_weight pretrain` | `../out/pretrain_768.pth` | `../out/pretrain_768_moe.pth` |
| SFT 输入 | `--from_weight pretrain` | `../out/pretrain_768.pth` | `../out/pretrain_768_moe.pth` |
| SFT 输出 | `--save_weight ocean_sft_replay` | `../out/ocean_sft_replay_768.pth` | `../out/ocean_sft_replay_768_moe.pth` |
| DPO/GRPO 输入 | `--from_weight ocean_sft_replay` | `../out/ocean_sft_replay_768.pth` | `../out/ocean_sft_replay_768_moe.pth` |

因此 Dense 和 MoE 会自动分开加载、保存，不会互相覆盖。架构必须贯穿整条链路：MoE SFT 必须增加 `--use_moe 1`，并从 MoE Pretrain 权重开始；后续 MoE DPO/GRPO 同理。训练 checkpoint 另存于 `../checkpoints/`，并带相同的 `_moe` 后缀和 `_resume` 续训后缀。

## 四卡烟测

所有训练命令从 `trainer/` 运行。先以 SwanLab offline 模式各跑 20 个 optimizer steps：

```bash
cd /home/anhuang/OceanHeart/trainer

torchrun --standalone --nproc_per_node=4 train_pretrain.py \
  --data_path ../data/processed/pretrain_train.jsonl \
  --val_data_path ../data/processed/pretrain_val.jsonl \
  --max_steps 20 --eval_interval 10 --save_interval 10 \
  --use_swanlab --swanlab_mode offline --run_name smoke-pretrain

torchrun --standalone --nproc_per_node=4 train_full_sft.py \
  --data_path ../data/processed/ocean_sft_replay_train.jsonl \
  --val_data_path ../data/processed/ocean_sft_val.jsonl \
  --max_steps 20 --eval_interval 10 --save_interval 10 \
  --use_swanlab --swanlab_mode offline --run_name smoke-sft

torchrun --standalone --nproc_per_node=4 train_dpo.py \
  --max_steps 20 --eval_interval 10 --save_interval 10 \
  --use_swanlab --swanlab_mode offline --run_name smoke-dpo
```

续训检查：先用 `--max_steps 10` 运行，再用相同参数加 `--from_resume 1 --max_steps 20`；同一个 SwanLab run 应从第 11 个 optimizer step 继续。

## 完整实验

### 1. 通用预训练

#### Dense Pretrain

```bash
torchrun --standalone --nproc_per_node=4 train_pretrain.py \
  --epochs 2 --dtype float16 --batch_size 16 --accumulation_steps 4 \
  --max_seq_len 340 --learning_rate 5e-4 \
  --data_path ../data/processed/pretrain_train.jsonl \
  --val_data_path ../data/processed/pretrain_val.jsonl \
  --use_swanlab --swanlab_project OceanHeart-Pretrain --run_name baseline-pretrain
```

若 11GB 显存 OOM，只改为 `--batch_size 8 --accumulation_steps 8`，有效 batch 不变。

#### MoE Pretrain

MoE 使用 4 experts、top-1 路由，总参数约 198.4M，每个 token 激活约 63.9M。`--use_moe 1` 会自动保存为 `pretrain_768_moe.pth`：

```bash
torchrun --standalone --nproc_per_node=4 train_pretrain.py \
  --use_moe 1 --save_weight pretrain \
  --epochs 2 --dtype float16 --batch_size 8 --accumulation_steps 8 \
  --max_seq_len 340 --learning_rate 5e-4 \
  --data_path ../data/processed/pretrain_train.jsonl \
  --val_data_path ../data/processed/pretrain_val.jsonl \
  --use_swanlab --swanlab_project OceanHeart-Pretrain --run_name baseline-pretrain-moe
```

### 2. 海洋 SFT 对照

Dense 的两次训练必须使用同一个 `pretrain_768.pth`、seed 和参数，只改变数据：

```bash
torchrun --standalone --nproc_per_node=4 train_full_sft.py \
  --save_weight ocean_sft_pure --from_weight pretrain \
  --data_path ../data/processed/ocean_sft_train.jsonl \
  --val_data_path ../data/processed/ocean_sft_val.jsonl \
  --epochs 2 --dtype float16 --batch_size 2 --accumulation_steps 16 \
  --max_seq_len 768 --learning_rate 1e-5 \
  --use_swanlab --swanlab_project OceanHeart-SFT --run_name ocean-sft-pure

torchrun --standalone --nproc_per_node=4 train_full_sft.py \
  --save_weight ocean_sft_replay --from_weight pretrain \
  --data_path ../data/processed/ocean_sft_replay_train.jsonl \
  --val_data_path ../data/processed/ocean_sft_val.jsonl \
  --epochs 2 --dtype float16 --batch_size 2 --accumulation_steps 16 \
  --max_seq_len 768 --learning_rate 1e-5 \
  --use_swanlab --swanlab_project OceanHeart-SFT --run_name ocean-sft-replay
```

MoE SFT 使用相同数据对照和超参数，但两条命令都增加 `--use_moe 1`；它会自动加载 `pretrain_768_moe.pth` 并保存带 `_moe` 后缀的结果：

```bash
torchrun --standalone --nproc_per_node=4 train_full_sft.py \
  --use_moe 1 --save_weight ocean_sft_pure --from_weight pretrain \
  --data_path ../data/processed/ocean_sft_train.jsonl \
  --val_data_path ../data/processed/ocean_sft_val.jsonl \
  --epochs 2 --dtype float16 --batch_size 2 --accumulation_steps 16 \
  --max_seq_len 768 --learning_rate 1e-5 \
  --use_swanlab --swanlab_project OceanHeart-SFT --run_name ocean-sft-pure-moe

torchrun --standalone --nproc_per_node=4 train_full_sft.py \
  --use_moe 1 --save_weight ocean_sft_replay --from_weight pretrain \
  --data_path ../data/processed/ocean_sft_replay_train.jsonl \
  --val_data_path ../data/processed/ocean_sft_val.jsonl \
  --epochs 2 --dtype float16 --batch_size 2 --accumulation_steps 16 \
  --max_seq_len 768 --learning_rate 1e-5 \
  --use_swanlab --swanlab_project OceanHeart-SFT --run_name ocean-sft-replay-moe
```

若 MoE SFT 在 11GB 显存上 OOM，改为每卡 `--batch_size 1 --accumulation_steps 32`，保持有效 batch 不变。

#### SFT Pure vs Replay 评估

Pure 曲线更短不是 early stopping：当前训练代码没有开启 early stop。Pure 有 40,152 条样本，Replay 有 44,614 条；在 4 卡、每卡 batch 2、累积 16、训练 2 epochs 时，理论上分别约为 628 和 698 个 optimizer steps。

正式对比位于 [`SFT Evaluation`](./SFT%20Evaluation/README.md)，固定比较 Pretrain（SFT 前基线）、Pure 和 Replay，并复用现有数据完成：

| 评估面 | 指标 | 用途 |
|---|---|---|
| Ocean SFT test | assistant-token NLL / PPL | 海洋领域拟合 |
| Generic SFT eval | assistant-token NLL / PPL | 灾难性遗忘 |
| 5 个现有多领域 benchmark | chat-template 长度归一化候选准确率 | 海洋、通用中英文和科学推理迁移 |
| 固定海洋 prompts | greedy 生成、吞吐、可选 SiliconFlow 裁判 | 实际回答质量与风格 |

```bash
/home/anhuang/.conda/envs/minimind/bin/python \
  "SFT Evaluation/evaluator.py" --architecture dense

/home/anhuang/.conda/envs/minimind/bin/python \
  "SFT Evaluation/evaluator.py" --architecture moe
```

快速检查可加 `--max-samples 5 --generation-samples 2`。评估输出 Pure vs Replay 的逐样本差值、2,000 次配对 bootstrap 95% 区间和 Markdown/CSV 汇总；现有 `eval_ocean.py` 继续作为跨 Pretrain/SFT/DPO/GRPO 的轻量全流程体检入口。

注意：当前 Pure 与 Replay 是相同 epoch，而 Replay 数据多约 11%，所以不是严格固定 token 预算的单变量实验。若要严谨归因，应再补一组相同 `max_steps` 的训练。

### 3. 海洋 DPO

```bash
torchrun --standalone --nproc_per_node=4 train_dpo.py \
  --from_weight ocean_sft_replay --save_weight ocean_dpo \
  --epochs 2 --dtype float16 --batch_size 1 --accumulation_steps 4 \
  --max_seq_len 1024 --learning_rate 4e-8 --beta 0.15 \
  --use_swanlab --swanlab_project OceanHeart-DPO --run_name ocean-dpo
```

MoE DPO 使用同一命令，增加 `--use_moe 1`，并把 run name 改为 `--run_name ocean-dpo-moe`；它会自动加载 `ocean_sft_replay_768_moe.pth` 并保存 `ocean_dpo_768_moe.pth`。

### 4. 海洋 LoRA 对照

LoRA 与全量 SFT 从同一个 `pretrain` 权重出发，使用相同的 90% OceanInstruct + 10% 通用回放数据。默认 rank 16，仅训练注意力的 `q_proj/o_proj`（约 0.393M 参数）。训练会保存适配器 `ocean_lora_768.pth` 和可直接评估的合并权重 `ocean_lora_merged_768.pth`。

```bash
torchrun --standalone --nproc_per_node=4 train_lora.py \
  --from_weight pretrain --save_weight ocean_lora --lora_rank 16 \
  --data_path ../data/processed/ocean_sft_replay_train.jsonl \
  --val_data_path ../data/processed/ocean_sft_val.jsonl \
  --epochs 2 --dtype float16 --batch_size 2 --accumulation_steps 16 \
  --max_seq_len 768 --learning_rate 1e-4 \
  --use_swanlab --swanlab_project OceanHeart-LoRA --run_name ocean-lora
```

MoE LoRA 使用同一命令，增加 `--use_moe 1`，并把 run name 改为 `--run_name ocean-lora-moe`；它会自动加载 `pretrain_768_moe.pth`，适配器和合并权重分别保存为 `ocean_lora_768_moe.pth`、`ocean_lora_merged_768_moe.pth`。

### 5. 海洋 GRPO（首轮 20 steps）

GRPO 默认从 Dense `ocean_sft_replay` 开始。奖励由 SiliconFlow 海洋领域裁判给出：每个问题把参考答案和 4 个候选合并为一次请求，再叠加小幅长度奖励与三元组重复惩罚。训练过程不会保存或记录 API Key；API 连续三次失败或返回非法分数会直接停止，避免用伪造的零奖励污染实验。

```bash
# 只在自己的终端设置新密钥，不要写入脚本或提交到 Git
export SILICONFLOW_API_KEY='你的新密钥'

torchrun --standalone --nproc_per_node=4 train_grpo.py \
  --from_weight ocean_sft_replay --save_weight ocean_grpo \
  --data_path ../data/processed/ocean_grpo_train.jsonl \
  --val_data_path ../data/processed/ocean_grpo_val.jsonl \
  --epochs 1 --dtype float16 --batch_size 1 --accumulation_steps 1 \
  --max_seq_len 768 --max_gen_len 256 --num_generations 4 \
  --learning_rate 3e-7 --beta 0.1 --epsilon 0.2 --max_steps 20 \
  --use_swanlab --swanlab_project OceanHeart-GRPO --run_name ocean-grpo-20steps
```

4 卡、每卡 batch 1 时，20 steps 约产生 80 次训练裁判请求；默认的启动验证与两次周期验证各最多 2 batch，另有少量请求。费用和耗时以 SiliconFlow 当时的模型计费为准。

MoE GRPO 使用同一命令，增加 `--use_moe 1`，并把 run name 改为 `--run_name ocean-grpo-moe`；它会自动加载 `ocean_sft_replay_768_moe.pth`。由于 policy、reference 和 rollout 同时占用显存，2080 Ti 上先将 `--max_steps` 设为 2 做烟测，再决定是否长跑。

## 离线评估

```bash
cd /home/anhuang/OceanHeart
python eval_ocean.py
```

结果写入 `artifacts/eval/summary.json`、`generations.jsonl` 和 `generations.csv`，默认比较 Pretrain、合并 LoRA、纯/回放 SFT、DPO、GRPO，包括 Ocean/Generic NLL 与 PPL、DPO 偏好准确率和 margin、固定问题的确定性回答及吞吐。

可选开启 SiliconFlow 裁判：

```bash
SILICONFLOW_API_KEY=... python eval_ocean.py --judge siliconflow
```

没有密钥时只跳过 Judge，不影响离线指标。实验结论填写到 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 许可与引用

- 模型和训练代码派生自 MiniMind，遵循 Apache License 2.0，详见 `LICENSE` 与 `NOTICE`。
- OceanInstruct-v0.2 遵循其 MIT 数据许可，使用时请引用 OceanGPT：*OceanGPT: A Large Language Model for Ocean Science Tasks*, arXiv:2310.02031。
