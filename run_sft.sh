#!/bin/bash
cd /root/autodl-tmp/target_1/LLaMA-Factory
LOG_FILE="../train_ocean_qwen.log"

# 指定使用哪张显卡 (0代表第一张)
export CUDA_VISIBLE_DEVICES=0

echo "开始训练 Ocean Qwen.., (加入 --quantization_bit 4)."

nohup llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path /root/autodl-tmp/Qwen2.5-7B-Instruct \
    --dataset ocean_data \
    --template qwen \
    --finetuning_type lora \
    --lora_target all \
    --quantization_bit 4 \
    --output_dir /root/autodl-tmp/checkpoints/ocean_qwen_sft_v1 \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 1.0 \
    --max_samples 4000 \
    --val_size 0.1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --logging_steps 10 \
    --save_steps 200 \
    --plot_loss \
    --fp16 \
    > $LOG_FILE 2>&1 &

echo "训练已在后台启动！"
echo "数据条:4000 "
echo "请使用 'tail -f $LOG_FILE' 查看实时日志。"
