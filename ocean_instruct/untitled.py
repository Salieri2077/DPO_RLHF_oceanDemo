import json
from datasets import load_dataset
import os

# ================= 路径配置 (请确认无误) =================
# LlamaFactory 的 data 根目录
LF_DATA_DIR = "/root/autodl-tmp/target_1/LLaMA-Factory/data"

# 1. 原始数据目录 (你下载的数据)
SOURCE_DATA_PATH = "/root/autodl-tmp/target_1/ocean_instruct/"

# 2. 目标数据保存路径 (ocean_data.json)
TARGET_DATA_PATH = os.path.join(LF_DATA_DIR, "ocean_data.json")

# 3. 配置文件路径 (dataset_info.json) <--- 这里修正了！
INFO_FILE_PATH = os.path.join(LF_DATA_DIR, "dataset_info.json")
# =======================================================

print("🚀 步骤 1: 开始处理并保存数据...")

# 确保目标目录存在
os.makedirs(LF_DATA_DIR, exist_ok=True)

try:
    # 尝试加载 HuggingFace 格式目录
    ds = load_dataset(SOURCE_DATA_PATH, split="train")
except Exception as e:
    print(f"⚠️ 目录加载失败，尝试寻找 json 文件... ({e})")
    # 尝试寻找目录下的 json 文件
    json_files = [f for f in os.listdir(SOURCE_DATA_PATH) if f.endswith('.json')]
    if json_files:
        full_path = os.path.join(SOURCE_DATA_PATH, json_files[0])
        print(f"👉 找到文件: {full_path}")
        ds = load_dataset("json", data_files=full_path, split="train")
    else:
        raise FileNotFoundError("无法在源目录找到任何 .json 数据文件！")

# 保持 ShareGPT 格式直接保存
output_data = [item for item in ds]

with open(TARGET_DATA_PATH, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"✅ 数据已保存至: {TARGET_DATA_PATH}")


print("\n🚀 步骤 2: 修改 dataset_info.json 注册表...")

# 定义配置内容
new_dataset_config = {
    "file_name": "ocean_data.json",
    "formatting": "sharegpt", 
    "columns": {
        "messages": "conversations"
    },
    "tags": {
        "role_tag": "from",
        "content_tag": "value",
        "user_tag": "human",
        "assistant_tag": "gpt"
    }
}

# 读取并更新 dataset_info.json
current_config = {}
if os.path.exists(INFO_FILE_PATH):
    try:
        with open(INFO_FILE_PATH, 'r', encoding='utf-8') as f:
            current_config = json.load(f)
    except json.JSONDecodeError:
        print("⚠️ 原配置文件损坏，将新建。")
else:
    print("⚠️ 未找到 dataset_info.json，将新建。")

# 插入配置
current_config["ocean_data"] = new_dataset_config

# 保存
with open(INFO_FILE_PATH, 'w', encoding='utf-8') as f:
    json.dump(current_config, f, ensure_ascii=False, indent=2)

print(f"✅ 注册成功！配置已写入: {INFO_FILE_PATH}")
print("🎉 一切就绪！现在可以运行 llamafactory-cli train 命令了！")