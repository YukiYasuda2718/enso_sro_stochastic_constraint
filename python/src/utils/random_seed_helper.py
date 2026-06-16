import os
import random
from logging import getLogger

import numpy as np
import torch

logger = getLogger()


def set_all_seeds(seed: int = 42) -> None:
    try:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        torch.use_deterministic_algorithms(True)

    except Exception as e:
        logger.error(e)
