import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Bool, Float, Int

def softmax(in_features: Float[Tensor, " ..."], dim: int):
    max_val = torch.max(in_features, dim=dim, keepdim=True).values
    in_features = in_features - max_val

    y = torch.exp(in_features)
    sums = torch.sum(y, dim=dim, keepdim=True)

    y = y / sums
    return y



def cross_entropy(logits, targets):
    '''
        logits: Float[Tensor, " batch_size vocab_size"]
        targets: Int[Tensor, " batch_size"]
    '''
    batch_size = targets.shape[-1]
    # 需要使用 Log-Sum-Exp 技巧，参考数学公式

    # 找出每一行的最大值
    max_val = torch.max(logits, dim=-1, keepdim=True).values    # [batch_size, 1]
    print(max_val.shape)

    # 找到targets对应的 logits
    selected_logits = logits[torch.arange(batch_size), targets]

    logits = logits - max_val      # [batch_size, vocab_size]
    logits = torch.exp(logits)

    expsum = torch.sum(logits, dim=-1, keepdim=True)     # [batch_size, 1] 
    logexpsum = torch.log(expsum)  # [batch_size, 1] 

    loss = max_val + logexpsum - selected_logits       # [batch_size, 1]

    return loss.mean()
