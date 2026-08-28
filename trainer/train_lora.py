import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.lm_dataset import SFTDataset
from trainer.train_lm import run_lm_training


if __name__ == "__main__":
    run_lm_training(
        SFTDataset,
        "OceanHeart LoRA fine-tuning",
        {
            "stage": "LoRA",
            "save_weight": "ocean_lora",
            "batch_size": 2,
            "accumulation_steps": 16,
            "learning_rate": 1e-4,
            "max_seq_len": 768,
            "eval_interval": 100,
            "data_path": "../data/processed/ocean_sft_replay_train.jsonl",
            "val_data_path": "../data/processed/ocean_sft_val.jsonl",
            "from_weight": "pretrain",
            "project": "OceanHeart-LoRA",
        },
        use_lora=True,
    )
