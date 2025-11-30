import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
from tqdm import tqdm
import re
import ast

# --- 配置 ---
CANDIDATE_FILE = "dpo_candidates_raw.json"
# CANDIDATE_FILE = "dpo_candidates_raw_0.json"

FINAL_DPO_FILE = "ocean_dpo_data.json"
BASE_MODEL_PATH = "/root/autodl-tmp/Qwen2.5-7B-Instruct" # 裁判模型路径

class Judge:
    def __init__(self):
        # 量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.eval()

    def rank_responses(self, prompt, candidates):
        """
        Listwise Ranking: 一次性对比所有候选，返回排名索引。
        """
        # 1. 动态构建 Input 文本
        candidates_text = ""
        for i, resp in enumerate(candidates):
            candidates_text += f"\n[Response {i}]:\n{resp}\n"

        # 2. 裁判 Prompt
        judge_prompt = f"""
[Task]
You are an expert judge in Ocean Science.
I will provide a User Question and {len(candidates)} Assistant Responses.
Your goal is to RANK the responses from Best to Worst based on accuracy, helpfulness, and depth.

[User Question]: 
{prompt}

{candidates_text}

[Instruction]
1. Analyze the pros and cons of each response briefly.
2. Output the ranking as a Python list of indices, e.g., [2, 0, 1] means Response 2 is best, Response 1 is worst.
3. STRICT FORMAT: The last line must be the list only.

[Output]
Reasoning:
<your analysis>
Ranking:
"""
        messages = [{"role": "user", "content": judge_prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer([text], return_tensors="pt", padding=True).to(self.model.device)

        with torch.no_grad():
            gen_ids = self.model.generate(
                inputs.input_ids, 
                attention_mask=inputs.attention_mask,
                max_new_tokens=512, # 需要稍微长一点来容纳多个回答的点评
                temperature=0.2    # 低温，保证逻辑严谨
            )
        
        output = self.tokenizer.decode(gen_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        
        # 3. 解析结果：寻找列表 [x, x, x]
        try:
            # 使用正则找到最后出现的列表结构
            matches = re.findall(r"\[([\d,\s]+)\]", output)
            if matches:
                # 取最后一个匹配项（防止前面Reasoning里也有列表）
                last_match = matches[-1]
                # 转为 Python 列表
                ranking = [int(x.strip()) for x in last_match.split(',')]
                
                # 校验：确保包含的索引合法
                if len(ranking) == len(candidates) and set(ranking) == set(range(len(candidates))):
                    return ranking, output
        except:
            pass
            
        return None, output

def main():
    with open(CANDIDATE_FILE, 'r', encoding='utf-8') as f:
        items = json.load(f)
    
    print(f"正在加载裁判模型 (Listwise Ranking Mode)...")
    judge = Judge()
    
    dpo_pairs = []
    
    print(f"开始处理 {len(items)} 条数据...")
    for item in tqdm(items):
        prompt = item['prompt']
        candidates = item['candidates'] # 这是一个包含 3 个字符串的列表
        
        if len(candidates) < 2:
            continue

        # 1. 让裁判排序
        ranking, raw_output = judge.rank_responses(prompt, candidates)
        
        if ranking:
            # ranking[0] 是第一名的索引 (Best)
            # ranking[-1] 是最后一名的索引 (Worst)
            best_idx = ranking[0]
            worst_idx = ranking[-1]
            
            best_resp = candidates[best_idx]
            worst_resp = candidates[worst_idx]
            
            # 2. 构造 DPO 数据
            # 只有当最好和最坏不是同一个内容时才保存（防重）
            if best_resp != worst_resp:
                dpo_pairs.append({
                    "conversations": [
                        {"from": "human", "value": prompt}
                    ],
                    "chosen": {"from": "gpt", "value": best_resp},
                    "rejected": {"from": "gpt", "value": worst_resp}
                })
            else:
                print("警告：最好和最坏的回答内容完全一致，跳过。")
        else:
            print(f"解析失败或格式错误。Prompt前20字: {prompt[:20]}")
            # print(f"Output片段: {raw_output[-100:]}") # 调试用

    # 保存
    with open(FINAL_DPO_FILE, 'w', encoding='utf-8') as f:
        json.dump(dpo_pairs, f, ensure_ascii=False, indent=2)
        
    print(f"\n处理完成！")
    print(f"原始数据: {len(items)}")
    print(f"生成 DPO 样本: {len(dpo_pairs)}")
    print("下一步：在 LlamaFactory 中将 dataset_info.json 注册该文件，并开启 Stage: dpo 训练。")

if __name__ == "__main__":
    main()
