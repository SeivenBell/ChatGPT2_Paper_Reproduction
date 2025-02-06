import torch
from dataclasses import dataclass
import torch.nn as nn
from torch.nn import functional as F

# ------------------------------------------------------------------------------


@dataclass  # decorator in Python is used to automatically generate special methods for classes,
# such as __init__, __repr__, __eq__, __hash__, and __str__.


class GPTConfig:
    """
    Configuration class for the GPT model.

    Attributes:
        block_size (int): The size of each block in the model.
        vocab_size (int): The size of the vocabulary.
        n_layers (int): The number of layers in the model.
        n_head (int): The number of attention heads.
        n_mod (int): The dimension of the model.
    """
    block_size: int = 256
    vocab_size: int = 65
    n_layers: int = 6
    n_head: int = 6
    n_mod: int = 384


class GPT(nn.Module):
    def _init_(self, config):
        super().__init__()
        self.config = config
