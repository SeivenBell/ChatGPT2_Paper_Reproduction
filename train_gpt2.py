import torch
from dataclasses import dataclass
import torch.nn as nn
from torch.nn import functional as F

# ------------------------------------------------------------------------------


@dataclass  # decorator in Python is used to automatically generate special methods for classes,
# such as __init__, __repr__, __eq__, __hash__, and __str__.


class GPTConfig:
    """
    GPTConfig is a configuration class for setting up the parameters of a GPT-2 model.

    Attributes:
        block_size (int): The size of each block in the model. Default is 256.
        vocab_size (int): The size of the vocabulary. Default is 65.
        n_layers (int): The number of layers in the model. Default is 6.
        n_head (int): The number of attention heads in each layer. Default is 6.
        n_mod (int): The dimension of the model. Default is 384.
    """
    # Strong Typing in Python here
    block_size: int = 256
    vocab_size: int = 65
    n_layers: int = 6
    n_head: int = 6
    n_mod: int = 384


class GPT(nn.Module):
    """
    GPT model class inheriting from PyTorch's nn.Module.

    Args:
        config (GPTConfig): Configuration object for the GPT model.
    """
    def _init_(self, config):
        """
        Initializes the GPT model with the given configuration.

        Args:
            config (GPTConfig): Configuration object for the GPT model.
        """
        super().__init__()
        self.config = config
