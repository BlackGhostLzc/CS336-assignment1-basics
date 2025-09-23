import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Bool, Float, Int
from .linear import Linear


def sigmoid(x):
    o = 1/(1 + torch.exp(-x))  
    return o


def Silu(x):
    return x * torch.sigmoid(x)



class Swiglu(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        '''
        up_proj:   w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        down_proj: w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        geted:     w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.
        '''
        self.up_proj = Linear(d_in=d_model, d_out=d_ff)
        self.down_proj = Linear(d_in=d_ff, d_out=d_model)
        self.gated = Linear(d_in=d_model, d_out=d_ff)
    
    def forward(self, in_features: Float[Tensor, " ... d_model"]):
        u = self.up_proj(in_features)
        g = self.gated(in_features)
        o = (Silu(u)) * g
        return self.down_proj(o)
    
    def init_weights(self, w1: Float[Tensor, " d_ff d_model"], 
                     w2: Float[Tensor, "d_model d_ff"], 
                     w3: Float[Tensor, " d_ff d_model"]):
        self.up_proj.init_weights(w1)
        self.down_proj.init_weights(w2)
        self.gated.init_weights(w3)


