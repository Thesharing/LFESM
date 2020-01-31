import random
import torch
import numpy as np


def seed_all(num: int = 42):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(num)
        torch.backends.cudnn.deterministic = True
