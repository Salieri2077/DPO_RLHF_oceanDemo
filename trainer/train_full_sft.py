import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.lm_dataset import SFTDataset
from trainer.train_lm import run_lm_training


if __name__ == "__main__":
    run_lm_training(
        SFTDataset,
        "OceanHeart supervised fine-tuning",
        {
            "stage": "SFT",
            "save_weight": "ocean_sft_replay",
            "batch_size": 2,
            "accumulation_steps": 16,
            "learning_rate": 1e-5,
            "max_seq_len": 768,
            "eval_interval": 100,
            "data_path": "../data/processed/ocean_sft_replay_train.jsonl",
            "from_weight": "pretrain",
            "project": "OceanHeart-SFT",
        },
    )
