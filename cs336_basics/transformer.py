import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Bool, Float, Int
from .linear import Linear
from .attention import *
from .rmsnorm import RMSNorm
from .activations import Swiglu, Silu
from .embedding import Embedding

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
        self.mha.init_weights(weights["attn.q_proj.weight"], weights["attn.k_proj.weight"], 
                               weights["attn.v_proj.weight"], weights["attn.output_proj.weight"])
        
        self.pre_rms.init_weights(weights["ln1.weight"])
        self.post_rms.init_weights(weights["ln2.weight"])

        self.swiglu.init_weights(weights["ffn.w1.weight"], weights["ffn.w2.weight"], weights["ffn.w3.weight"])


    def init_weights_2(self, wq, wk, wv, wo, ln1, ln2, ffnw1, ffnw2, ffnw3):
        self.mha.init_weights(wq, wk, wv, wo)
        self.pre_rms.init_weights(ln1)
        self.post_rms.init_weights(ln2)
        self.swiglu.init_weights(ffnw1, ffnw2, ffnw3)



class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, 
                 num_layers, num_heads, d_ff, rope_theta):
        super().__init__()

        self.embedding = Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        for _ in range(num_layers):
            block = TransformerBlock(d_model, num_heads, d_ff, context_length,rope_theta)
            self.layers.append(block)

        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_in=d_model, d_out=vocab_size)


    def forward(self, x):
        '''
            x: Int[Tensor, " batch_size sequence_length"],
        '''
        x = self.embedding(x)    # [batch_size, seq_len, d_model]

        for _, layer in enumerate(self.layers):
            x = layer(x)
        
        x = self.ln_final(x)
        x = self.lm_head(x)

        return x


    def init_weights(self, weights):
        self.embedding.init_weights(weights["token_embeddings.weight"])

        for i in range(0, self.num_layers):
            # 每一层的 QKV
            self.layers[i].init_weights_2(weights[f"layers.{i}.attn.q_proj.weight"],
                                        weights[f"layers.{i}.attn.k_proj.weight"],
                                        weights[f"layers.{i}.attn.v_proj.weight"],
                                        weights[f"layers.{i}.attn.output_proj.weight"],
                                        weights[f"layers.{i}.ln1.weight"],
                                        weights[f"layers.{i}.ln2.weight"],
                                        weights[f"layers.{i}.ffn.w1.weight"],
                                        weights[f"layers.{i}.ffn.w2.weight"],
                                        weights[f"layers.{i}.ffn.w3.weight"])
            
            
        self.ln_final.init_weights(weights["ln_final.weight"])
        self.lm_head.init_weights(weights["lm_head.weight"])


        