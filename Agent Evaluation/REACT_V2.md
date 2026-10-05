# OceanHeart ReAct v2：运行环境与工具 SFT

初始版本独立实现，参考 Hello-Agents 的 ReAct 行动—观察循环及 Claude Agent SDK 的运行边界思想。
目前默认执行循环已改为 **OpenAI Agents SDK**，详见 [SDK 接入说明](AGENTS_SDK.md)。
模型、海洋工具、数据协议与 SFT 监督不变；旧循环只用于历史复现/回归。
MiniMind 架构与 Apache-2.0 署名保持不变。本轮**没有 Agentic RL**。

## 使用

环境：激活 `/home/anhuang/.venvs/ocean-agents-sdk`（安装见 SDK 文档），从仓库根目录执行。生成产物与权重不提交 Git。

```bash
source /home/anhuang/.venvs/ocean-agents-sdk/bin/activate
# 只生成一次，存在 manifest 时拒绝覆盖；变更协议应使用新目录。
python scripts/prepare_ocean_react.py --output data/processed/react-v2
python -m unittest discover -s tests

# 短程 SFT 单独命令（输出名必须未使用）
python -m torch.distributed.run --standalone --nproc_per_node=4 trainer/train_agent.py \
  --agent_version v2 --mode sft --data_dir data/processed/react-v2 \
  --from_weight ocean_agent_dense-agent-20260927-0232-aligned-sft \
  --save_weight ocean_react_v2_sft --run_name react-v2-sft \
  --epochs 3 --max_steps 300 --max_train_seconds 3600 \
  --accumulation_steps 8 --learning_rate 1e-5 --eval_interval 50 \
  --use_swanlab --swanlab_project OceanHeart-Agent

# 完整实验：四卡烟测/恢复/云端核验 → A/B基线 → 独立SFT → C与锁定测试
python scripts/run_ocean_react.py --tag react-v2-YYYYMMDD-HHMM

# 交互（将权重路径替换为实际最佳权重；首期不提供WebUI）
python scripts/chat_ocean_react.py --weight out/ocean_react_v2_sft_best_768.pth \
  --session artifacts/agent/chat-example.json --interactive
# 仅旧 local 会话支持进程中断恢复；SDK 目前支持同进程多轮交互，不支持 crash-resume。
python scripts/chat_ocean_react.py --runtime local --weight out/ocean_react_v2_sft_best_768.pth \
  --session artifacts/agent/chat-example.json --resume --interactive
```

CLI 也支持 `--question '调查船以12 km/h航行3 h，航程是多少？'`。默认使用训练侧语料库；可以用 `--corpus` 选择另一个已准备的本地资料库。交互澄清等待真实用户输入，不自动补充答案。上下文已满时应开启新会话，不偷偷删掉先前消息。

## 执行协议与边界

模型可以输出 `<plan>一句行动说明</plan>`，然后选择一个 `<tool_call>{"name":...,"arguments":...}</tool_call>`、`<final>答案</final>` 或 `<clarify>待确认信息</clarify>`。旧模型不带标记的最终答案也能解析。行动说明不是强制长推理，不被评分为“推理能力”。

仅开放本地 SQLite FTS5 检索与明确公式的海洋计算；无 Bash、文件修改、网络请求或任意代码执行。工具资料不可信，不能添加权限。模型参数错误只返回错误，不由 Harness 修正为标准答案。

默认最多6轮生成、4次调用尝试（包括非法调用），每轮192 tokens，完整输入加预留输出最多2048 tokens，单次工具观察最多256 tokens。成功调用重复一次会收到警告，再次重复终止；暂时性错误仅允许一次同参数重试。所有错误和终止原因入轨迹。`valid_calls`/`tool_valid_rate`沿用旧统计口径：**返回无错误的调用比例**，包含执行可用性，不是纯JSON语法准确率。

SDK 保存轨迹和工具边界快照，但目前不支持恢复中断进程。旧 local 的恢复快照只在完整轮次边界保存；已保存轮次不会重复执行，进程若恰在工具完成而快照尚未落盘时崩溃，该未提交轮次可能重做。当前仅只读工具，未承诺对有外部写入的工具做到 exactly-once。CLI 一个会话文件同时只由一个进程使用。

## 数据与监督

3000训练／150验证／300锁定测试，每个集合六类等量：计算、证据摘录、检索-单位换算-计算、错误恢复、缺参/无资料/冲突澄清、直接照录已给出的资料。

OceanInstruct继承原始资料划分，并排除旧Agent基准使用的来源；问题、数值任务组合、资料源与措辞模板族跨集合隔离。各集合仍覆盖相同六种技能，**措辞留出不是未见任务类别泛化**。合成卡、冲突记录和暂时故障均明确为练习，不能当真实海洋观测或航行决策。

所有示范通过真实执行验证。错误历史只作为上下文；SFT只监督后续正确的助手输出，工具观察不计损失。临时故障任务由隔离的模拟环境注入一次故障；真实推理默认没有该故障。评分标准保留在数据任务的私有字段，运行时只得到问题、公开历史以及环境反馈。

“无需工具”主要考察提供资料后的忠实转述，并不代表完整海洋知识问答；澄清按缺失字段/冲突关键词及禁止编造数值的规则检查，不宣称覆盖所有自然语言语义。短摘录仍是证据操作实验，OceanInstruct本身可能有合成或错误内容。

## 对照与验收

|阶段|权重|Harness|评估|
|---|---|---|---|
|A|旧工具SFT|v1|旧50题回归|
|B|同一旧工具SFT|v2 SDK|旧回归、新验证；训练结束后新测试|
|C|新SFT最佳验证权重|v2 SDK|与B配对的相同集合和预算|

上述是后续启动的默认流程；已完成的 `react-v2-20260927-1046` 历史实验采用自建循环，记录不追溯改写。

测试不用于选权重。最佳按150题贪心验证成功率选择，持平保留更早权重（可能就是初始权重）；最后权重单独保留。测试另以42/43/44固定种子每题各采样一次，不是best-of-three；报告单次采样均值及按题聚类的配对bootstrap区间。工程只运行一次训练seed，不能据此宣称跨训练seed稳定。

SwanLab项目 `OceanHeart-Agent`：SFT、烟测、各评估分run；云端烟测续训沿用同一个run。旧SwanLab版本的offline恢复会新建一个run并记录来源ID，不把两个进程的offline目录误认为两个rank重复写入。

每个run在 `artifacts/agent/<run-name>` 保存配置、JSONL实际轨迹、CSV和汇总；流水线根目录保存 `status.json`、各阶段日志、`comparison.json` 和逐题 `comparison.csv`。状态明确区别运行中、失败和完成。`comparison.json`中分别记录Harness变化、SFT变化和测试配对差异；不同数据集的绝对成功率不能直接对比。

本轮不根据loss或reward正负宣称能力提高，也不因测试退化自动重训。下一阶段Agentic RL须另行确认。
