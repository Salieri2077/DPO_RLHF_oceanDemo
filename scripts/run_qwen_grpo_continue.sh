#!/bin/bash
# Continue the Qwen3-1.7B GRPO run from step 200 to 400 (2026-10-09). The first 200 steps used a cosine schedule that had
# decayed to 1e-6, so this leg re-warms to 1e-5 over 10 steps, holds it, and decays over the last 50 steps (WSD).
# Then SFT vs GRPO@200 vs GRPO@400 are judged together by DeepSeek-V3 on the same 100 test questions.
set -e  # no pipefail: torchrun may segfault at exit after saving; completion.json is the success check
cd /home/anhuang/OceanHeart
PY=/home/anhuang/.conda/envs/minimind/bin/python
TAG=qwen3-1.7b-ocean-grpo-q72b-20261008-2318
ADAPTER=out/hf/${TAG//-/_}
OUT="GRPO Evaluation/results/$TAG-step400"
export SILICONFLOW_API_KEY=$(cat ~/.siliconflow_key)
mkdir -p "$OUT"
rm -f artifacts/grpo_hf/$TAG/completion.json
cd trainer
OMP_NUM_THREADS=4 /home/anhuang/.conda/envs/minimind/bin/torchrun --standalone --nproc_per_node=4 train_grpo_hf.py \
  --save_adapter ${TAG//-/_} --run_name $TAG --resume --max_steps 400 --schedule wsd --schedule_start 200 \
  --warmup_steps 10 --decay_steps 50 --learning_rate 1e-5 --num_generations 8 --temperature 0.8 --max_new_tokens 1024 \
  --beta 0.04 --lora_rank 32 --reward_model Qwen/Qwen2.5-72B-Instruct \
  --val_questions 16 --val_generations 4 --eval_interval 25 --save_interval 25 \
  --use_swanlab --swanlab_mode cloud --swanlab_project OceanHeart-GRPO \
  2>&1 | grep --line-buffered -v -i -E 'warn|pynvml' >> ../logs/$TAG.log
cd ..
test -f artifacts/grpo_hf/$TAG/completion.json
sleep 30
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k $PY "GRPO Evaluation/evaluate_qwen_grpo.py" --stage generate --adapter grpo200=${ADAPTER}_step200 \
    --adapter grpo400=${ADAPTER}_final --output "$OUT" --shard $k --num_shards 4 > "$OUT/log_gen_$k.txt" 2>&1 &
done
wait
$PY "GRPO Evaluation/evaluate_qwen_grpo.py" --stage judge --adapter grpo200=${ADAPTER}_step200 --adapter grpo400=${ADAPTER}_final \
  --output "$OUT" > "$OUT/log_judge.txt" 2>&1
echo DONE > "$OUT/chain_done"
