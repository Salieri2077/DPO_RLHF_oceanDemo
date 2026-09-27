# ReAct v2 启动验收记录

这是启动记录，不是训练完成或效果提升报告。

- 分支：`codex/ocean-react-v2`；实现 commit：`97ef0f7`。
- tmux：`ocean-react-dense`。
- 流水线：`react-v2-20260927-1046`；配置为四卡全参数工具 SFT，**没有 Agentic RL**。
- [正式 SFT SwanLab](https://swanlab.cn/@Salieri2077/OceanHeart-Agent/runs/1lrm9oc7ppryagk1cz60d)。
- [四卡烟测／续训 SwanLab](https://swanlab.cn/@Salieri2077/OceanHeart-Agent/runs/dhol4pkd5atmheaa40yv6)。

## 已核验

- 全仓26个测试通过。新增检查覆盖数据划分、标准答案不进入运行时、工具权限、重复动作、调用限制、错误恢复、会话轮次恢复、SFT精确前缀/错误历史屏蔽及有限反向传播。
- 两卡参数同步、checkpoint保存与恢复成功，global step从2增加到3。旧版SwanLab的offline恢复采用新run并保留来源ID；每次启动只有rank0写入。
- 四卡烟测5次更新均为有限非零梯度，模型参数哈希跨卡一致；恢复至第6次，云端run ID不变。
- 烟测5次更新共18.80秒（约3.76秒/次，包含每步保存和同步验收）。正式SFT纯更新窗口实测约0.54–0.60秒/次，不包括周期验证与保存。
- 正式SFT从原始工具SFT重新开始，未继承烟测权重；交接时已实际完成至少50次更新。
- SwanLab OpenApi读取正式run的`train/loss`与`train/grad_norm`返回HTTP 200；最后核验的loss为0.02671086、梯度范数为0.72262782。训练loss不是Agent能力成绩。
- CLI已用真实Dense权重完成计算工具调用与会话恢复测试。

## 起点与基线

权重：`out/ocean_agent_dense-agent-20260927-0232-aligned-sft_768.pth`。
SHA-256：`d81faa0d9a89dbfc8d37c7e0b290e72ca76b4bfe1cd2d27a9171406c7d706fc2`。

|对照|集合|成功率|
|---|---|---|
|A：旧SFT＋旧Harness|旧50题|86%|
|B：同权重＋新Harness|旧50题|84%|
|B：同权重＋新Harness|新150题|4%|

这些结果已保留，不能宣称新Harness单独改善了能力；新任务与旧任务百分比不可直接比较。新SFT的C组结果尚待生成。

数据：3000训练／150验证／300锁定测试，六类均衡；示范真实执行验收，未静默截断。初版开发数据保存在`data/processed/react-v2-preflight-20260927/`，正式数据为`data/processed/react-v2/`，未删除旧数据。

## 后续自动流程与位置

SFT最多3 epochs、300步或1小时，按先到为准；当前3000样本配置跑满3epochs为282次更新。按固定新验证集选最佳权重，持平保留早期权重。之后自动执行C组回归/验证、B/C锁定测试、三固定seed的采样配对评估；不因结果差自动重训。

- 总日志：`logs/react_v2_pipeline_20260927_1046.log`。
- 阶段状态及日志：`artifacts/agent/react-v2-20260927-1046/`。
- 正式SFT轨迹/指标：`artifacts/agent/react-v2-20260927-1046-sft/`。
- 最后／最佳权重：`out/ocean_react-v2-20260927-1046-sft_768.pth`、`out/ocean_react-v2-20260927-1046-sft_best_768.pth`。
- 最终报告：流水线目录下的`comparison.json`、`comparison.csv`；阶段未完成时报告可能只有已有基线，以`status.json`为准。
- 使用方式见[ReAct v2说明](REACT_V2.md)。

交接后不继续实时监测。最终效果及失败案例等训练和评估结束后再分析。
