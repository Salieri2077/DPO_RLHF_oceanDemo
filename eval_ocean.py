import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_path = "/root/autodl-tmp/Qwen2.5-7B-Instruct"
lora_path = "/root/autodl-tmp/checkpoints/ocean_qwen_sft_v1" 

tokenizer = AutoTokenizer.from_pretrained(base_path)
model = AutoModelForCausalLM.from_pretrained(base_path,device_map = 'auto',torch_dtype=torch.float16)load_in_8bit=True
model = PeftModel.from_pretrained(model,lora_path)

questions = [
    "台风对海滩前滨地形的主要影响是什么？",
    "简述海洋盐度分布的规律。",
    "什么是赤潮，它是如何形成的？"
]

print("-" * 30)
for q in questions:
    messages = [
        {"role": "system", "content": "你是海洋科学专家 Ocean Qwen。"},
        {"role": "user", "content": q}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
        
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = output.split("assistant")[-1].strip()
    
    print(f"Q: {q}")
    print(f"A: {response}\n")
    print("-" * 30)

