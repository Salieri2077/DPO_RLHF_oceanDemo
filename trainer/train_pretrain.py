import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.lm_dataset import PretrainDataset
from trainer.train_lm import run_lm_training


if __name__ == "__main__":
    run_lm_training(
        PretrainDataset,
        "OceanHeart pretraining",
        {
            "stage": "Pretrain",
            "save_weight": "pretrain",
            "batch_size": 16,
            "accumulation_steps": 4,
            "learning_rate": 5e-4,
            "max_seq_len": 340,
            "eval_interval": 500,
            "data_path": "../data/processed/pretrain_train.jsonl",
            "from_weight": "none",
            "project": "OceanHeart-Pretrain",
        },
    )
