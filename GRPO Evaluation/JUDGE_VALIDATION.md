# 2026-09-22 裁判验收

模型：SiliconFlow `Qwen/Qwen3-32B`，temperature=0，非思考模式，JSON输出。
代码提供真实API验收入口 `validate_judge.py`，原始报告在忽略提交的 `results/judge32b_gate.json`。

第一次请求发现两候选返回四个分数，三次重试均失败。明确本次候选数量后重新完整验收：6组用例，正反序各重复2次，共24次成功评分请求。

| 用例 | 原顺序分数 | 反序还原后 | 结果 |
|---|---|---|---|
| 历史答案 id 0、1、20、44（4组） | [-3,-3] | [-3,-3] | 两轮均通过 |
| 完全相同正确答案 | [3,3] | [3,3] | 两轮均通过 |
| 两个好答案、重复废话、错误解释 | [3,2,-2,-3] | [3,3,-2,-3] | 两轮均未通过位置一致性 |

门槛在实测前固定：换序分差不超过0.5。最后一组有1分偏差，因此总验收失败。没有启动新GRPO训练，没有修改原始分数让测试通过。

旧的严重评分翻转在这些回归用例中已消失，但32B仍存在位置偏差。需要进一步改进评分协议或更换裁判，再完整重跑验收。候选模型的通过用例不能证明全域可靠性，也不能证明训练会获得收益。

## 同提示词扩大裁判模型复测

新增 `--model` 参数，允许显式选择模型而不改变训练默认值。通过账户 `/v1/models` 核实模型标识；以下两款各完成24次真实评分，原有0.5分容差不变。

| API 模型 | 通过的用例轮次（共12轮） | 最大换序分差 | 验收 |
|---|---:|---:|---|
| Qwen/Qwen3-32B（此前基线） | 10 | 1.0 | 失败 |
| Qwen/Qwen2.5-72B-Instruct | 10 | 1.0 | 失败 |
| deepseek-ai/DeepSeek-V3 | 8 | 1.0 | 失败 |

72B四候选用例中，好答案从3分变为2分，第二轮重复废话从-2变为-3。DeepSeek-V3对历史id 0的同一候选换序后从-3变为-2，四候选用例也出现1分换序差异。所有模型的完全相同正确答案均获[3,3]。

原始报告：`results/judge72b_gate.json`、`results/judge_v3_gate.json`，包含候选、每轮正反序分数、提示词全文及哈希。结果支持“本次换用更大模型未消除位置偏差”，不支持“参数量没有价值”的一般结论，因为模型代际、架构也同时改变。没有启动训练或将未通过的新模型设为默认。

复现示例：

```bash
python "GRPO Evaluation/validate_judge.py" --model Qwen/Qwen2.5-72B-Instruct --output "GRPO Evaluation/results/judge72b_gate.json"
python "GRPO Evaluation/validate_judge.py" --model deepseek-ai/DeepSeek-V3 --output "GRPO Evaluation/results/judge_v3_gate.json"
```

## 用户接受偏差后的训练决策（2026-09-22）

用户明确接受本次已观察到的1分偏差，指定 DeepSeek-V3 用于新一轮 GRPO。验收报告仍保留失败结果，不改动0.5分门槛。先启动 Dense，MoE 待后续启动；均从各自 SFT replay 起点重新训练，而非续接旧 GRPO。

Dense 命令（在 trainer 目录、minimind 环境运行，密钥由环境继承）：

```bash
torchrun --standalone --nproc_per_node=4 train_grpo.py \
  --from_weight ocean_sft_replay --from_resume 0 --use_moe 0 \
  --save_weight ocean_grpo_deepseekv3_eval50 \
  --data_path ../data/processed/ocean_grpo_train.jsonl \
  --val_data_path ../data/processed/ocean_grpo_val.jsonl \
  --epochs 1 --dtype float16 --batch_size 1 --accumulation_steps 1 \
  --max_seq_len 768 --max_gen_len 256 --num_generations 4 \
  --learning_rate 3e-7 --beta 0.1 --epsilon 0.2 --max_steps 100 \
  --eval_interval 10 --eval_batches 50 --reward_model deepseek-ai/DeepSeek-V3 \
  --use_swanlab --swanlab_mode cloud --swanlab_project OceanHeart-GRPO \
  --run_name ocean-grpo-dense-deepseekv3-100steps-eval50
```

裁判和提示词均与旧8B实验不同，reward绝对值不能直接作为模型能力提升量；训练后须用统一裁判配对评估。
