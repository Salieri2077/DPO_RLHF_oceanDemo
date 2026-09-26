# OceanHeart Dense Agent 实验

基于 MiniMind（Apache-2.0）架构，复用 OceanHeart 的训练基础设施。原始 Dense GRPO 是起点；工具 SFT 和 Agent GRPO 是两个独立训练阶段，不把冷启动收益全部归因于 RL。

## 环境与数据

使用 `/home/anhuang/.conda/envs/minimind`，不安装新框架。两个工具分别是 SQLite FTS5 本地海洋检索和受限航速／距离／时间计算。工具不执行任意 Python、SQL、shell 或网络请求；SQLite 查询有执行时限，工具返回最多 256 tokens。每条轨迹最多三次生成、两次调用。

OceanInstruct-v0.2 为 MIT 数据，包含合成内容且可能有误；本实验不把资料当作权威事实。航次卡是明确标识的合成练习，非实测数据。训练／验证／测试任务数量 2000／50／50；计算、检索摘录、检索后计算比例 40／40／20。语料库按原始数据集合隔离，属于开放资料工具评估。

```bash
cd /home/anhuang/OceanHeart
/home/anhuang/.conda/envs/minimind/bin/python scripts/prepare_ocean_agent.py
/home/anhuang/.conda/envs/minimind/bin/python -m unittest tests.test_agent
```

生成任务、完整工具示范和 manifest 位于忽略提交的 `data/processed/`。数据准备阶段实际执行每条示范的检索，验证目标原文可见且整条示范不超长。

## 运行与门槛

在 tmux 中启动以下控制程序（tag 必须唯一）：

```bash
OMP_NUM_THREADS=2 /home/anhuang/.conda/envs/minimind/bin/python scripts/run_ocean_agent.py \
  --tag dense-agent-YYYYMMDD-HHMM --hours 10
```

流程：固定 50 题原始验证、20 题可选 V3 评估、32 题×4 次诊断；必要时工具 SFT（3 epochs／200 updates／一小时上限，以先到为准），再诊断。门槛：调用合法率≥80%、完整成功率≥20%、有组内奖励差异的题目≥20%，且三类任务均有成功记录。诊断使用固定训练侧题目，不是泛化成绩。

门槛失败就停止；通过后四卡 GRPO 烟测5步＋续训1步，核验梯度、参数哈希同步、保存和恢复；正式 GRPO 从通过诊断的起点重新开始，不继承烟测更新。模型和冻结 reference 均使用这个 RL 起点。

总预算包含原始能力检查、可选裁判评估、SFT、诊断和烟测。训练截止前预留30分钟保存与末尾评估。停止在 optimizer 边界，因此耗时是近似上限，而非强制 kill 的硬截止。显存不足、非有限梯度、SwanLab 写入异常或门槛失败均停止，不能自动放宽条件。数据、工具或关键训练参数变化时拒绝续训。

## 奖励与可解释性

每条轨迹只各计一次：相关合法工具0.2、正确中间结果0.3、完成所有必要工具步骤并给出正确答案0.5；未恢复错误和未完成轨迹−1。计算答案核验数值（绝对误差1e-3或相对误差1e-4）和单位。检索要求返回目标原文及正确引用；这不是开放式知识问答的语义准确率。

逐轮保存实际采样的输入和输出，只在该轮生成 tokens 上算 policy/KL；工具结果作为上下文，不作为预测标签。生成 `temperature=1, top_p=1, top_k=0` 与计算概率口径一致。每轨迹先按有效生成 token 数归一化，再平均轨迹。原有单轮 GRPO 默认行为不变。

SwanLab 项目为 `OceanHeart-Agent`。阶段分 run，训练记录任务成功、合法调用、分项奖励、组内差异、policy loss、KL、梯度、长度和吞吐。验证用同一50题贪心生成，训练用每题4条随机轨迹：两者的采样口径不同，不直接比较 reward 的绝对水平。

`artifacts/agent/<tag>/status.json` 记录流水线阶段及终止原因；阶段目录保存配置、CSV、JSON、逐轮轨迹、SwanLab 本地记录。`out/` 保留 SFT、最终和最佳验证权重；`checkpoints/` 保留优化器、FP32参数、scaler、RNG、global step 和 run ID。不同 run 使用独立名称，旧权重不覆盖。

DeepSeek V3 仅评分固定20道检索验证题，读取 `SILICONFLOW_API_KEY`。无密钥、超时或解析错误记录为缺失，不能作为低分。规则训练不依赖 API。评估入口：

```bash
/home/anhuang/.conda/envs/minimind/bin/python 'Agent Evaluation/evaluate_judge.py' \
  --traces artifacts/agent/实际运行名/val_final.jsonl \
  --output artifacts/agent/实际运行名/judge_final.json
```

## 实验记录模板

| 阶段 | 权重／commit／SwanLab | 计算成功率 | 检索成功率 | 两工具成功率 | 合法调用率 | V3辅助分数 |
|---|---|---|---|---|---|---|
| 原始 Dense GRPO | | | | | | |
| 工具 SFT（如执行） | | | | | | |
| Agent GRPO 最佳／最终 | | | | | | |

另记录：诊断是否通过、零优势组占比、实际训练步数／时长、失败轨迹与终止原因。SFT后→RL后的变化才是本轮RL的增量证据；原始→SFT后的变化单独列出。未改善或未通过门槛也属于结果，不通过 loss 正负宣称有效。
