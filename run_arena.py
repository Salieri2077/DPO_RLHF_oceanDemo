import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd
from tqdm import tqdm
import gc
import json
import os

# --- 配置 ---
BASE_MODEL = "/root/autodl-tmp/Qwen2.5-7B-Instruct"
SFT_ADAPTER = "/root/autodl-tmp/checkpoints/ocean_qwen_sft_v1"
DPO_ADAPTER = "/root/autodl-tmp/checkpoints/ocean_qwen_dpo_v1"

# 指向你的 DPO 数据文件
TEST_DATA_JSON = "/root/autodl-tmp/target_1/LLaMA-Factory/data/ocean_dpo_data.json"
OUTPUT_FILE = "model_battle_results.xlsx"

# 采样 20 条做对比
NUM_SAMPLES = 2

def load_dpo_data(file_path, limit=20):
    print(f"正在读取 DPO 数据: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 截取前 N 条
    data = data[:limit]
    
    parsed_data = []
    for item in data:
        # 提取 DPO 格式中的 Prompt 和 Chosen 答案
        prompt = item['conversations'][0]['value']
        chosen = item['chosen']['value']
        
        parsed_data.append({
            "query": prompt,
            "reference_chosen": chosen # 把此作为参考答案放入 Excel
        })
    return parsed_data

def generate_responses(model_path, adapter_path, data_list, tag):
    print(f"\n>>> [阶段] 正在加载模型: {tag} ...")
    print(f"    Base: {model_path}")
    print(f"    Adapter: {adapter_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    
    # 4-bit 加载基座
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True,
        trust_remote_code=True
    )
    
    # 加载 Adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    responses = []
    print(f"    正在生成推理 (共 {len(data_list)} 条)...")
    
    for item in tqdm(data_list):
        query = item['query']
        messages = [
            {"role": "system", "content": "你是一个海洋科学领域的专家助手。"},
            {"role": "user", "content": query}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer([text], return_tensors="pt", padding=True).to(model.device)
        
        with torch.no_grad():
            gen_ids = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=512,
                temperature=0.7, 
                top_p=0.9
            )
        
        output = tokenizer.batch_decode(gen_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        responses.append(output)
        
    # 彻底清理显存，防止加载下一个模型时 OOM
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print("    显存已清理。")
    
    return responses

def main():
    # 1. 准备数据
    data_list = load_dpo_data(TEST_DATA_JSON, limit=NUM_SAMPLES)
    df = pd.DataFrame(data_list)
    
    # 2. 跑 SFT 模型
    sft_responses = generate_responses(BASE_MODEL, SFT_ADAPTER, data_list, "SFT_Model")
    df['SFT_Response'] = sft_responses
    
    # 3. 跑 DPO 模型
    dpo_responses = generate_responses(BASE_MODEL, DPO_ADAPTER, data_list, "DPO_Model")
    df['DPO_Response'] = dpo_responses
    
    # 4. 整理列顺序，方便查看
    # 顺序：问题 -> SFT回答 -> DPO回答 -> (参考)DPO训练时的好回答
    cols = ['query', 'SFT_Response', 'DPO_Response', 'reference_chosen']
    df = df[cols]
    
    # 5. 保存
    print(f"\n正在保存对比报告至 {OUTPUT_FILE}...")
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"完成！请下载 {OUTPUT_FILE} 查看 SFT vs DPO 的区别。")

if __name__ == "__main__":
    main()