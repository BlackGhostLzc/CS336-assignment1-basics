import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Bool, Float, Int
from .linear import Linear
import math

class Rope(nn.Module):
    def __init__(self, d_k, max_seq_len, theta):
        super().__init__()
        assert d_k % 2 == 0, "In Rope, d_k must be an even number"

        self.d_k = d_k
        indices = torch.arange(0, self.d_k, 2, dtype=torch.float32) # [0, 2, 4 ... d-2]
        exponent = -indices / self.d_k
        theta = torch.pow(theta, exponent)          # [theta0, theta1, theta2, ..... theta(d//2-1)]

        multipliers = torch.arange(0, max_seq_len)  # [0, 1, 2, ... max_seq_len-1]
        theta = torch.outer(multipliers, theta)     # 计算外积
        '''
            0*theta0, 0*theta1, 0*theta2, ............. 0*theta(d//2-1)
            1*theta0, 1*theta1, 1*theta2, ............. 1*theta(d//2-1)
            .............
            (max_seq_len-1)*theta0
        '''
        
        sin = torch.sin(theta)
        cos = torch.cos(theta)

        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)


    
    def forward(self, x, token_positions: None):
        '''
            x: [... sequence_length d_k]，比如说 [... num_heads, seq_len, head_dim]
        '''
        seq_len = x.shape[-2]

        x0 = x[..., 0::2]   # (0, 2, 4, 6, 8 ...)   [... num_heads, seq_len, head_dim//2]
        x1 = x[..., 1::2]   # (1, 3, 5, 7, 9 ...)   [... num_heads, seq_len, head_dim//2]

        '''
            o0 = x0 * cos(m*theta0) - x1 * sin(m*theta0)
            01 = x1 * cos(m*theta0) + x0 * sin(m*theta0)

            o2 = x2 * cos(m*theta1) - x3 * sin(m*theta1)
            03 = x3 * cos(m*theta1) + x2 * sin(m*theta1)
            
        '''
        '''
            cosMatrix sinMatrix: [seq_len, head_dim//2]
            [
               cos(0*theta0)  cos(0*theta1)  cos(0*theta2)  ... cos(0*theta(d_k/2-1))   第一个位置
               cos(1*theta0)  cos(1*theta1)  cos(1*theta2)  ... cos(1*theta(d_k/2-1))   第二个位置
               ....
               cos((max_seq_len-1)*theta0)   ..............
            ]
            这两个matrix可以预先计算出来, 存入buffer cache里面, 避免重复的计算
        '''
        # 逐元素相乘
        # cosMatrix = self.cos[:seq_len, :]
        # sinMatrix = self.sin[:seq_len, :]
        if token_positions is not None:
            cosMatrix = self.cos[token_positions]
            sinMatrix = self.sin[token_positions]
        else: 
            cosMatrix = self.cos[:seq_len, :]
            sinMatrix = self.sin[:seq_len, :]

        o0 = x0 * cosMatrix - x1 * sinMatrix 
        o1 = x1 * cosMatrix + x0 * sinMatrix

        # 逐元素堆叠
        output = torch.stack([o0, o1], dim=-1)      
        output = torch.flatten(output, start_dim=-2)
        return output



