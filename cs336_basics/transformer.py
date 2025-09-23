import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Bool, Float, Int
from .linear import Linear
from .attention import *
from .rmsnorm import RMSNorm
from .activations import Swiglu, Silu

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta):
        super().__init__()

        eps = 1e-5
        self.mha = MultiHeadSelfAttention(d_model, num_heads, True, max_seq_len, theta)
        self.pre_rms =  RMSNorm(d_model, eps)
        self.post_rms = RMSNorm(d_model, eps)
        self.swiglu = Swiglu(d_model, d_ff)


    def forward(self, x):
        residual = x
        x = self.pre_rms(x)
        x = self.mha(x)
        x = residual + x

        residual = x
        x = self.post_rms(x)
        x = self.swiglu(x)
        x = residual + x
        return x
        

    
    def init_weights(self, weights):
        self.mha.init_werights(weights["attn.q_proj.weight"], weights["attn.k_proj.weight"], 
                               weights["attn.v_proj.weight"], weights["attn.output_proj.weight"])
        
        self.pre_rms.init_weights(weights["ln1.weight"])
        self.post_rms.init_weights(weights["ln2.weight"])

        self.swiglu.init_weights(weights["ffn.w1.weight"], weights["ffn.w2.weight"], weights["ffn.w3.weight"])


