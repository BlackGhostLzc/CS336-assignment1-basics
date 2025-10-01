import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Bool, Float, Int
import torch.nn.init as init
import math

class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.weight = nn.Parameter(torch.randn(d_out, d_in))

        # 2. 根据公式计算标准差 (std)
        std = math.sqrt(2 / (d_in + d_out))
        
        # 3. 使用 trunc_normal_ 进行原地初始化
        # a 是下界, b 是上界
        init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)



    def forward(self, x: Tensor):
        '''
            x: [..., d_in]
        '''
        output = x @ self.weight.T
        return output
    
    def init_weights(self, w: Float[Tensor, " d_ff d_model"]):
        self.weight.data.copy_(w)
    
