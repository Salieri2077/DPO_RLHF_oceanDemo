# OceanHeart + OpenAI Agents SDK

The default v2 runtime replaces the handwritten ReAct loop with `agents.Runner`.
`agent/sdk.py` implements the SDK `Model` interface and registers the existing
`search_ocean` and `marine_calculate` functions as SDK `FunctionTool`s. Inference
still uses MiniMind locally; no OpenAI model, API key, server, or paid API is used.
MiniMind's Apache-2.0 attribution and OceanInstruct's dataset attribution remain unchanged.

## Install without changing the training environment

From the repository root:

```bash
/home/anhuang/.conda/envs/minimind/bin/python -m venv --system-site-packages /home/anhuang/.venvs/ocean-agents-sdk
/home/anhuang/.venvs/ocean-agents-sdk/bin/pip install -r requirements-agent-sdk.txt
```

The tested SDK is 0.22.3 (OpenAI client 3.19.2). Torch/Transformers are reused
from MiniMind. The original environment's OpenAI 1.59.6 is not upgraded.
Use this SDK environment for v2 commands. Legacy v1 keeps its local runtime;
`--runtime local` explicitly selects the old v2 loop for historical replay.

## Run the local backbone

```bash
OMP_NUM_THREADS=2 /home/anhuang/.venvs/ocean-agents-sdk/bin/python scripts/chat_ocean_react.py \
  --runtime sdk \
  --weight out/ocean_react-v2-20260927-1046-sft_best_768.pth \
  --session artifacts/agent/my-sdk-session.json --interactive
```

Use a fresh session path. `--question '...'` is available for one-shot runs.
Clarification follow-ups in the same process preserve the actual conversation.
SDK crash-resume and streaming are deliberately unsupported; `--resume` fails
explicitly. The existing `--runtime local` retains its original resume support.
Completed SDK traces and tool-boundary snapshots are saved as local JSON.

## Evaluate through the existing trainer

```bash
OMP_NUM_THREADS=2 /home/anhuang/.venvs/ocean-agents-sdk/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 trainer/train_agent.py \
  --runtime sdk --agent_version v2 --mode eval \
  --from_weight ocean_react-v2-20260927-1046-sft_best \
  --run_name my-sdk-val50 --data_dir data/processed/react-v2 --split val --limit 50
```

Results: `artifacts/agent/my-sdk-val50/result.{json,jsonl,csv}`. Use a distinct
run name for each comparison. Add the existing `--use_swanlab` options if cloud
evaluation metrics are desired. OpenAI tracing is separate: the adapter installs
an empty trace provider **process-wide**, and disables per-run tracing, so this
local-only process never constructs a cloud exporter. Do not embed it in a
process that needs other agents' OpenAI tracing.

## Contracts and boundaries

- SDK Runner, not `agent.react.run`, owns model/tool iteration. Our adapter keeps
  the checkpoint's original system prompt, tool schema and tokenizer template.
- Every SDK tool-call ID maps to the exact generated assistant text. Tool outputs
  are converted back to the original observation format. No answer correction,
  invented calls, forced successful arguments, or gold answers enter the model.
- Original token IDs and optional old log probabilities are saved per generation.
  Greedy CLI/evaluation deliberately save empty log probabilities. These traces
  must not be represented as ready-to-train GRPO samples.
- Malformed or unknown calls use an internal SDK error-feedback function which
  is not advertised in the model prompt and cannot execute a real tool.
- Six model rounds, four attempts, 2048 context tokens, 192 generated tokens and
  256 observation tokens are the defaults. No left truncation. Duplicate and
  retry limits match the existing harness; failures remain visible to the scorer.
- Retrieval keeps the existing split-specific corpus, parameter validation and
  untrusted-document boundary. Tool code, scoring and SFT datasets are reused.
- Trainer v2 **evaluation and SFT validation** now default to SDK. SFT's loss,
  dataset, assistant-token masks, optimizer and checkpoints are unchanged: its
  pre-training, periodic and final task evaluations use Runner. SDK GRPO/diagnose
  are still rejected; legacy v1 GRPO remains local. SDK rollout-based RL requires
  a separate sampling/weight-sync/probability verification step.
- Resume requires the same runtime, SDK version and adapter hash. Legacy
  checkpoints without a runtime are treated as local; pass `--runtime local` to
  resume them. To change runtimes, start a separately named run from its weights.
- One adapter per episode, synchronous CLI/worker entry. No concurrent reuse of
  one model/SQLite connection, no handoffs, hosted state, or structured-output API.

## Checks

```bash
OMP_NUM_THREADS=2 /home/anhuang/.venvs/ocean-agents-sdk/bin/python -m unittest discover -s tests -v
```

SDK tests cover actual Runner dispatch, exact multi-round token-prefix parity,
no network connections, all six task types, real local search/calculation,
error feedback, one-time outage recovery, clarification, repeated calls and
generation/context/turn/call limits. Optional SDK tests skip in the original
environment; existing tests continue to run there.

Official design references:
- https://developers.openai.com/api/docs/guides/agents/models
- https://developers.openai.com/api/docs/guides/agents/running-agents

## 2026-09-27 acceptance evidence

- Full suite in the SDK environment: 30 tests passed. In the original MiniMind
  environment the four optional SDK tests skip; no SDK dependency is imposed there.
- Two-rank GPU evaluation, same `ocean_react-v2-20260927-1046-sft_best` Dense
  checkpoint, greedy decoding, first 50 questions of the existing v2 validation
  split: SDK and local success both **20/50 (40%)**, 44 tool attempts each,
  tool validity 100%, average 1.88 generations/question.
- All 50 pairs exactly match on task ID, final answer, stop reason, calls/results,
  metrics and every round's input/completion token IDs. This is runtime parity,
  **not a model improvement claim**, and the v2 set is not the old 86% dataset.
- Successful real two-tool example: `val-0fc479497e2405b3`.
- The free-form CLI question asking for distance at 12 km/h for 3 h produced
  incorrect arguments twice and an error answer. This failure is retained, not
  repaired by the adapter or counted as a successful calculation.
- Raw evidence: `artifacts/agent/sdk-val50-check/result.{json,jsonl,csv}` and
  `artifacts/agent/local-val50-sdk-check/result.{json,jsonl,csv}`;
  CLI failure: `artifacts/agent/sdk-cli-check.json`.
- Logs: `logs/agent_sdk_tests.log`, `logs/agent_sdk_base_tests.log`,
  `logs/agent_sdk_val50.log`, `logs/agent_sdk_local_val50.log`, `logs/agent_sdk_cli.log`.
- No long training, cloud model calls, or SwanLab upload was started for this
  runtime acceptance check. Existing weights and datasets were not modified.

## Default-runtime migration and the existing SFT experiment

The old `ocean-react-dense` tmux pipeline was already **completed**, with an idle
shell and no training subprocess. It performed 282 SFT updates over 3 epochs;
the full pipeline, including evaluations, took about 21 minutes. Its output is
preserved at `artifacts/agent/react-v2-20260927-1046/`. No active job was killed,
no session was deleted, and no full training was restarted merely to change SDKs.

SDK now owns the default v2 interactive and evaluation loops. The existing
`scripts/run_ocean_react.py` pipeline also uses SDK for v2 SFT validation and
paired evaluations; only its explicitly labeled legacy A/v1 baseline remains
local. Run it with the SDK Python, a fresh tag, and only when another training
experiment is actually wanted. Existing trained weights need no SDK conversion.

工具 SFT 训练的是 **模型**，不是 SDK 或工具函数：学习何时检索、选择哪个工具、
正确填写数值与单位、读取真实结果并回答。SDK 提供调度机制，不自带我们的
OceanInstruct 分集合检索库或受限航程计算器。`search_ocean` 和 `marine_calculate`
继续复用同一份实现，已注册为 SDK FunctionTool；不重复写业务工具，也不换成
需要托管服务的通用搜索/代码执行工具。已有工具 SFT 数据与权重直接沿用。

`agent/react.py` retains shared protocol/scoring/dataset code and the historical
loop for regression tests/replay. The default SDK path does not call that loop.
Keeping the shared file unchanged also preserves existing data manifest hashes.

Default-switch checks: 31 tests pass in the SDK environment (original environment:
27 pass, 4 optional skips). A two-GPU engineering-only SFT smoke completed one
nonzero update, with parameter hashes synchronized. Both the initial and final
150-question validations contain SDK runtime traces. Resume reached global step 2
with synchronized parameters and finite nonzero gradients. Artifacts are isolated
under `artifacts/agent/sdk-default-sft-{smoke,resume}-20260927/`, with
`logs/agent_sdk_default_sft_{smoke,resume}.log`. SwanLab used **offline** mode for
these short checks; this is not a new full experiment or a claimed quality gain.
The saved production SFT checkpoint was not overwritten.
