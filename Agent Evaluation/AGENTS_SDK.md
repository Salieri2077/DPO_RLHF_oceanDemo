# OceanHeart + OpenAI Agents SDK

This optional runtime replaces the handwritten ReAct loop with `agents.Runner`.
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
- Trainer integration is **evaluation only** (`v2`); SDK SFT/GRPO/diagnose modes
  are rejected. Existing SFT/GRPO defaults remain unchanged. SDK rollout-based RL
  requires a separate sampling/weight-sync/probability verification step.
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
