import torch
from dataclasses import dataclass
import torch.nn as nn
from torch.nn import functional as F

# ------------------------------------------------------------------------------


@dataclass 

class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layers: int = 6
    n_head: int = 6
    n_mod: int = 384


class GPT(nn.Module):
    def _init_(self, config):
        super().__init__()
        self.config = config
