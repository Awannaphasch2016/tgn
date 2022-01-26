import random
import numpy as np
import torch

class Train:
    def __init__(self, args):
        self.args = args
    def set_random_seed(self):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

    def set_model_save_path(self):
        raise NotImplementedError

    def set_loggers(self):
        raise NotImplementedError

    def run_model(self):
        raise NotImplementedError
