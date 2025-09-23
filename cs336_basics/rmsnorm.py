import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Bool, Float, Int
from .linear import Linear


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-5):
        super().__init__()
        self.d_model = d_model
        # eps 加到分母上防止除 0, 稳定性
        self.eps = eps

        self.weight = nn.Parameter(torch.randn(self.d_model))


    def forward(self, x):
        '''
            x: Float[Tensor, " ... d_model"],
        '''
        denominator = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)   # [.... 1]
        x = (x / denominator) * self.weight
        return x
        

    def init_weights(self, weights):
        # weights: Float[Tensor, " d_model"]
        self.weight.data.copy_(weights)