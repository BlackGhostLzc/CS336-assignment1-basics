import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Bool, Float, Int

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = nn.Parameter(torch.randn(self.vocab_size, self.d_model))

    def forward(self, token_ids: Int[Tensor, " ..."]):
        return self.weight[token_ids]
    
    def init_weights(self, w: Float[Tensor, "vocab_size d_model"]):
        self.weight.data.copy_(w)