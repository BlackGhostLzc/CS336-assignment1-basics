import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Bool, Float, Int
import torch.nn.init as init

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = nn.Parameter(torch.randn(self.vocab_size, self.d_model))
        init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)


    def forward(self, token_ids: Int[Tensor, " ..."]):
        return self.weight[token_ids]
    
    
    def init_weights(self, w: Float[Tensor, "vocab_size d_model"]):
        self.weight.data.copy_(w)