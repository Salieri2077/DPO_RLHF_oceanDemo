# OceanHeart

OceanHeart 是一个基于 [MiniMind](https://github.com/jingyaogong/minimind) 训练骨架的 63.9M 参数海洋领域文本语言模型实验项目。目标不是隐藏上游实现，而是完整复现并分析：通用预训练 → 海洋 SFT → 海洋 DPO。

旧版 Qwen2.5-7B + LLaMA-Factory LoRA 工程保存在 Git 标签 `legacy-llamafactory-v1`；当前版本只保留可复现的数据、训练、评估主链。

## 环境与数据

使用已经验证的本地环境：

```bash
conda activate /home/anhuang/.conda/envs/minimind
cd /home/anhuang/OceanHeart
python scripts/prepare_ocean_data.py
```

数据处理会读取：

- `data/raw/ocean_instruct/OceanInstruct-v0.2.json`
- `data/raw/ocean_dpo_data.json`
- `/home/anhuang/minimind/dataset/pretrain_t2t_mini.jsonl`
- `/home/anhuang/minimind/dataset/sft_t2t_mini.jsonl`

生成文件位于 `data/processed/`，不会提交到 Git。`manifest.json` 记录数据来源、许可、稳定切分、过滤数量、token 长度和截断率。Ocean 数据按规范化问题哈希做 80/10/10 切分，DPO 与 SFT 的相同问题始终进入同一集合。

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

```bash
torchrun --standalone --nproc_per_node=4 train_pretrain.py \
  --epochs 2 --dtype float16 --batch_size 16 --accumulation_steps 4 \
  --max_seq_len 340 --learning_rate 5e-4 \
  --data_path ../data/processed/pretrain_train.jsonl \
  --val_data_path ../data/processed/pretrain_val.jsonl \
  --use_swanlab --swanlab_project OceanHeart-Pretrain --run_name baseline-pretrain
```

若 11GB 显存 OOM，只改为 `--batch_size 8 --accumulation_steps 8`，有效 batch 不变。

### 2. 海洋 SFT 对照

两次训练必须使用同一个 `pretrain_768.pth`、seed 和参数，只改变数据：

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

### 3. 海洋 DPO

```bash
torchrun --standalone --nproc_per_node=4 train_dpo.py \
  --from_weight ocean_sft_replay --save_weight ocean_dpo \
  --epochs 2 --dtype float16 --batch_size 1 --accumulation_steps 4 \
  --max_seq_len 1024 --learning_rate 4e-8 --beta 0.15 \
  --use_swanlab --swanlab_project OceanHeart-DPO --run_name ocean-dpo
```

## 离线评估

```bash
cd /home/anhuang/OceanHeart
python eval_ocean.py
```

结果写入 `artifacts/eval/summary.json`、`generations.jsonl` 和 `generations.csv`，包括 Ocean/Generic NLL 与 PPL、DPO 偏好准确率和 margin、固定问题的确定性回答及吞吐。

可选开启 SiliconFlow 裁判：

```bash
SILICONFLOW_API_KEY=... python eval_ocean.py --judge siliconflow
```

没有密钥时只跳过 Judge，不影响离线指标。实验结论填写到 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 许可与引用

- 模型和训练代码派生自 MiniMind，遵循 Apache License 2.0，详见 `LICENSE` 与 `NOTICE`。
- OceanInstruct-v0.2 遵循其 MIT 数据许可，使用时请引用 OceanGPT：*OceanGPT: A Large Language Model for Ocean Science Tasks*, arXiv:2310.02031。
