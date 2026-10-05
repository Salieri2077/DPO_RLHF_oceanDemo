# Pretrain Evaluation

Dense 与 MoE 共用同一 tokenizer、数据、随机种子和 candidate log-likelihood 协议。这里评估的是 Base Model，不套 chat template，也不让模型自由生成答案。

## Benchmark

| Domain | Benchmark | Split / 样本数 | 用途 |
|---|---|---:|---|
| General English | HellaSwag | validation / 500 | 英文常识与续写 |
| General Chinese | XCOPA-zh | test / 500 | 中文因果常识 |
| Science / Reasoning | ARC-Easy | test / 500 | 基础科学推理 |
| Ocean | OceanBenchmark Science_Text | train\* / 102 | 海洋科学选择题 |
| Ocean | MaritimeBench | test / 500 | 中文海事知识 |

\* OceanBenchmark 发布的 `Science_Text` 只有名为 `train` 的 102 条公开评测题；这里只把它当只读 benchmark，绝不并入训练。数据版本由 commit hash 固定，完整来源、许可、文件哈希和精确泄漏扫描结果见 `/data/anhuang/oceanheart_pretrain_eval/manifest.json`。

## 准备数据

新增数据只会写入 `/data/anhuang/oceanheart_pretrain_eval`：

```bash
cd "/home/anhuang/OceanHeart"
HF_ENDPOINT=https://huggingface.co \
  /home/anhuang/.conda/envs/minimind/bin/python \
  "Pretrain Evaluation/prepare_benchmarks.py"
```

默认用规范化 prompt 的精确 SHA-256 对 `/home/anhuang/minimind/dataset/pretrain_t2t_mini.jsonl` 做重合检查。该检查不能排除子串、改写或上游语料污染。

## 运行

一次运行两者并自动比较：

```bash
/home/anhuang/.conda/envs/minimind/bin/python \
  "Pretrain Evaluation/evaluator.py" --model all
```

也可以分别运行，第二次完成时会自动生成比较表：

```bash
/home/anhuang/.conda/envs/minimind/bin/python "Pretrain Evaluation/evaluator.py" --model dense
/home/anhuang/.conda/envs/minimind/bin/python "Pretrain Evaluation/evaluator.py" --model moe
```

默认 checkpoint 是 `out/pretrain_768.pth` 与 `out/pretrain_768_moe.pth`，序列长度 340。先做快速检查可加 `--max-samples 5 --device cpu`；正式比较不能给两者使用不同参数。

## 指标与输出

对每个选项计算 continuation token 的 log-likelihood，同时记录总和准确率和按 token 数归一化准确率；后者是主指标。还输出正确选项 NLL/PPL、prompt 截断率、参数量、checkpoint/tokenizer/manifest 哈希和耗时。

结果默认写入被 Git 忽略的 `Pretrain Evaluation/results/`：

- `dense|moe/predictions.jsonl`：每题原始分数；
- `dense|moe/summary.json`：按 benchmark/domain 汇总；
- `comparison.json/csv/md`：Dense vs MoE 差值和 2,000 次配对 bootstrap 95% 区间。

只有区间完全高于或低于 0 才报告 MoE 或 Dense 胜出，否则结论为 `Inconclusive`。PPL 是“正确候选 continuation PPL”，不是独立语料库的全文 PPL。

## 数据来源与许可

- [HellaSwag](https://github.com/rowanz/hellaswag)（MIT）
- [XCOPA](https://huggingface.co/datasets/cambridgeltl/xcopa)（CC-BY-4.0）
- [ARC](https://huggingface.co/datasets/allenai/ai2_arc)（CC-BY-SA-4.0）
- [OceanBenchmark](https://huggingface.co/datasets/zjunlp/OceanBenchmark)（MIT）
- [MaritimeBench](https://huggingface.co/datasets/Hi-Dolphin/MaritimeBench)（Apache-2.0）

运行最小测试：

```bash
cd "Pretrain Evaluation" && /home/anhuang/.conda/envs/minimind/bin/python -m unittest -v
```
