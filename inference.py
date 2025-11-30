import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
from tqdm import tqdm
import pandas as pd
import random
import os  # 引入 os 模块

# --- 配置 ---
DATA_PATH = "/root/autodl-tmp/target_1/LLaMA-Factory/data/ocean_data.json"
ADAPTER_PATH = "/root/autodl-tmp/checkpoints/ocean_qwen_sft_v1"
BASE_MODEL_PATH = "/root/autodl-tmp/Qwen2.5-7B-Instruct"
CANDIDATE_FILE = "dpo_candidates_raw.json"

# 采样数量
SAMPLE_SIZE = 400
# 每个 Prompt 生成几个回答
RESPONSES_PER_PROMPT = 3
# [新增] 每跑完多少条 Prompt 自动保存一次
SAVE_INTERVAL = 10 

def main():
    # 1. 加载 Prompt
    print(f"正在加载数据: {DATA_PATH}")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 随机抽取
    if len(data) > SAMPLE_SIZE:
        print(f"数据量 {len(data)} > {SAMPLE_SIZE}，进行随机采样...")
        data = random.sample(data, SAMPLE_SIZE)
    else:
        print(f"数据量 {len(data)}，将全量运行。")

    # 2. 加载模型 (4-bit)
    print("正在加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    # [修复1] 左填充
    tokenizer.padding_side = 'left' 
    
    # [修复2] 确保 pad_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    results = []
    
    # 检查是否有之前的进度（可选：如果你想断点续传，可以读取已存在的文件）
    # 这里为了简单，我们采用“覆盖写入”模式，但会实时更新文件

    print("开始生成...")
    # 3. 循环生成 (使用 enumerate 获取索引)
    for idx, item in enumerate(tqdm(data)):
        prompt = item['conversations'][0]['value']
        
        messages = [
            {"role": "system", "content": "你是一个海洋科学专家。"},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # [修复3] 显式开启 padding
        inputs = tokenizer(
            [text], 
            return_tensors="pt", 
            padding=True
        ).to(model.device)

        candidates = []
        
        # 连续生成
        for _ in range(RESPONSES_PER_PROMPT):
            with torch.no_grad():
                gen_ids = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,  # [关键修复]
                    pad_token_id=tokenizer.pad_token_id,   # [关键修复]
                    max_new_tokens=512,
                    temperature=0.9,    
                    top_p=0.9,
                    do_sample=True      
                )
            output = tokenizer.batch_decode(gen_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            candidates.append(output)
        
        results.append({
            "prompt": prompt,
            "candidates": candidates
        })

        # --- [新增] 自动保存逻辑 ---
        if (idx + 1) % SAVE_INTERVAL == 0:
            # 这里的 'w' 模式会覆盖写入整个 results 列表，起到更新作用
            # 这样如果崩了，直接读这个文件，里面就是最新的进度
            with open(CANDIDATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            # 在 tqdm 进度条旁边打印一个小提示（可选，不打扰进度条可注释掉）
            # tqdm.write(f"已自动保存进度: {idx + 1}/{len(data)}")

    # 4. 最终保存（防止最后一次不够 Interval 没存上）
    with open(CANDIDATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"全部完成！最终数据已保存至 {CANDIDATE_FILE}")

if __name__ == "__main__":
    main()