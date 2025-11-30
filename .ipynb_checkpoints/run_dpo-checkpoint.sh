#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

cd /root/autodl-tmp/target_1/LLaMA-Factory

echo "开始 DPO 训练..."

nohup llamafactory-cli train \
    --stage dpo \
    --do_train \
    --model_name_or_path /root/autodl-tmp/Qwen2.5-7B-Instruct \
    --adapter_name_or_path /root/autodl-tmp/checkpoints/ocean_qwen_sft_v1 \
    --dataset ocean_dpo \
    --template qwen \
    --finetuning_type lora \
    --lora_target all \
    --output_dir /root/autodl-tmp/checkpoints/ocean_qwen_dpo_v1 \
    --overwrite_output_dir \
    --quantization_bit 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --num_train_epochs 2.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --fp16 \
    --logging_steps 5 \
    --save_steps 50 \
    --plot_loss \
    --pref_beta 0.1 \
    --pref_ftx 0 \
    --pref_loss sigmoid \
    > ../train_dpo.log 2>&1 &

# 将上一个后台进程的 PID 存入变量
TRAIN_PID=$!

echo "DPO 训练已后台启动！PID 为 ${TRAIN_PID}，请查看 ../train_dpo.log"
echo "脚本将等待训练完成，然后执行关机操作。"

# -------------------------------------------------------------
# 关键步骤：等待训练进程结束，并根据结果决定是否关机
# -------------------------------------------------------------

# 1. 'wait ${TRAIN_PID}'：等待上面启动的训练进程结束
# 2. '&&'：如果 'wait' 成功（即训练进程返回状态码 0），则执行后面的命令
# 3. 'sudo poweroff'：执行关机命令（Autodl/大部分Linux系统推荐）

wait ${TRAIN_PID} && echo "✅ DPO 训练已完成！正在执行关机..." && sudo poweroff

# 如果你没有 sudo 权限，或者想用另一种关机方式，可以替换为：
# wait ${TRAIN_PID} && echo "✅ DPO 训练已完成！正在执行关机..." && shutdown -h now

# 如果训练失败 (wait 返回非 0 状态)，则脚本到此结束，不会执行关机命令。
# -------------------------------------------------------------