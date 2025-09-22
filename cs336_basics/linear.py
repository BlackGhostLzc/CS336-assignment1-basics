import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Bool, Float, Int

class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.weight = nn.Parameter(torch.randn(d_out, d_in))

    def forward(self, x: Tensor):
        '''
            x: [..., d_in]
        '''
        output = x @ self.weight.T
        return output
    
    def init_weights(self, w: Float[Tensor, " d_ff d_model"]):
        self.weight.data.copy_(w)
    
