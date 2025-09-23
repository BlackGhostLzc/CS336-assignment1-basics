import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Bool, Float, Int
from .linear import Linear
import math

from .rope import Rope


def scaled_dot_product_attention(Q, K, V, mask):
    '''
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None,
    '''
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-1, -2) / math.sqrt(d_k)  #[... queries, keys]
    if mask is not None:
        # mask 为 false 的地方置为负无穷
        scores = scores.masked_fill(~mask, float('-inf'))

    scores = torch.softmax(scores, dim=-1)
    output = scores @ V
    return output



class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, use_rope=False, max_seq_len: int | None = None,
                 theta: float | None = None, token_positions: torch.Tensor | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert self.d_model % self.num_heads == 0
        self.head_dim = self.d_model // self.num_heads

        self.WQ = Linear(self.d_model, self.d_model)
        self.WK = Linear(self.d_model, self.d_model)
        self.WV = Linear(self.d_model, self.d_model)
        self.WO = Linear(self.d_model, self.d_model)

        if use_rope == True:
            self.rope = Rope(d_k=self.head_dim, max_seq_len=max_seq_len, theta=theta)
        
        self.use_rope = use_rope
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.token_positions = token_positions
        

    
    def forward(self, x):
        '''
            x: [.... seq_len, d_model]
        '''
        original_shape = x.shape
        seq_len = x.shape[-2]              # ([4, 12, 64])

        q_proj = self.WQ(x)    # [.... seq_len, d_model]
        k_proj = self.WK(x)    # [.... seq_len, d_model]
        v_proj = self.WV(x)    # [.... seq_len, d_model]

        q_proj = q_proj.reshape(*original_shape[:-1], self.num_heads,-1)   # [.... seq_len, num_heads, d]
        k_proj = k_proj.reshape(*original_shape[:-1], self.num_heads,-1)   # [.... seq_len, num_heads, d]
        v_proj = v_proj.reshape(*original_shape[:-1], self.num_heads,-1)   # [.... seq_len, num_heads, d]

        q_proj = q_proj.transpose(-2, -3)   # [.... , num_heads, seq_len, d]
        k_proj = k_proj.transpose(-2, -3)   # [.... , num_heads, seq_len, d]
        v_proj = v_proj.transpose(-2, -3)   # [.... , num_heads, seq_len, d]
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

        if self.use_rope == True:
            q_proj = self.rope(q_proj, self.token_positions)
            k_proj = self.rope(k_proj, self.token_positions)

        output = scaled_dot_product_attention(q_proj, k_proj, v_proj, ~mask)    # [.... , num_heads, seq_len, d] 
        output = output.transpose(-2, -3)  # [.... ,  seq_len, num_heads, d]
        output = output.reshape(*original_shape[:-1], -1)  # [.... ,  seq_len, d_model]
        output = self.WO(output)
        return output




    def init_werights(self, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight):
        self.WQ.init_weights(q_proj_weight)
        self.WK.init_weights(k_proj_weight)
        self.WV.init_weights(v_proj_weight)
        self.WO.init_weights(o_proj_weight)