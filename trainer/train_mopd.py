"""Two-domain MOPD entry point for OceanHeart."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trainer.train_opd import build_parser, run_training


if __name__ == "__main__":
    run_training(build_parser(multi_domain=True).parse_args())
