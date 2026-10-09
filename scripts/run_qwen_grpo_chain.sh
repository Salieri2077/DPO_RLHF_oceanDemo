#!/bin/bash
# GRPO training -> paired SFT/GRPO generation on 4 GPUs -> DeepSeek-V3 judge. Launched 2026-10-08 as qwen3-1.7b-ocean-grpo-q72b-20261008-2318.
# The first attempt stopped at step 30 on SiliconFlow 429 rate limits; RESUME=1 continues from its state.pt.
set -e  # no pipefail: torchrun may segfault at exit after saving; completion.json is the success check
cd /home/anhuang/OceanHeart
PY=/home/anhuang/.conda/envs/minimind/bin/python
TAG=qwen3-1.7b-ocean-grpo-q72b-20261008-2318
OUT="GRPO Evaluation/results/$TAG"
export SILICONFLOW_API_KEY=$(cat ~/.siliconflow_key)
mkdir -p "$OUT"
cd trainer
OMP_NUM_THREADS=4 /home/anhuang/.conda/envs/minimind/bin/torchrun --standalone --nproc_per_node=4 train_grpo_hf.py \
  --save_adapter ${TAG//-/_} --run_name $TAG --max_steps 200 --num_generations 8 --temperature 0.8 --max_new_tokens 1024 \
  --learning_rate 1e-5 --beta 0.04 --lora_rank 32 --reward_model Qwen/Qwen2.5-72B-Instruct \
  --val_questions 16 --val_generations 4 --eval_interval 25 --save_interval 25 \
  --use_swanlab --swanlab_mode cloud --swanlab_project OceanHeart-GRPO ${RESUME:+--resume --swanlab_id latjgfyb5turdinm2uvd4} \
  2>&1 | grep --line-buffered -v -i -E 'warn|pynvml' >> ../logs/$TAG.log
cd ..
test -f artifacts/grpo_hf/$TAG/completion.json
sleep 30
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k $PY "GRPO Evaluation/evaluate_qwen_grpo.py" --stage generate --adapter out/hf/${TAG//-/_}_final \
    --output "$OUT" --shard $k --num_shards 4 > "$OUT/log_gen_$k.txt" 2>&1 &
done
wait
$PY "GRPO Evaluation/evaluate_qwen_grpo.py" --stage judge --adapter out/hf/${TAG//-/_}_final --output "$OUT" > "$OUT/log_judge.txt" 2>&1
echo DONE > "$OUT/chain_done"
