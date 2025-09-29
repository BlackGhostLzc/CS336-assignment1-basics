from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)


    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]           # Get the learning rate.

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                grad = p.grad.data     # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1     # Increment iteration number.
        
        return loss
    

# weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
# opt = SGD([weights], lr=1)
# for t in range(100):
#     opt.zero_grad()
#     # Reset the gradients for all learnable parameters.
#     loss = (weights**2).mean() # Compute a scalar loss value.
#     print(loss.cpu().item())
#     loss.backward() # Run backward pass, which computes gradients.
#     opt.step() # Run optimizer step.


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, amsgrad=False):
        '''
        1. params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        2. lr (float, optional): Learning rate (default: 1e-3).
        3. betas (Tuple[float, float], optional): Coefficients used for computing running averages
                of gradient and its square (default: (0.9, 0.999)).
        4. eps (float, optional): Term added to the denominator to improve numerical stability (default: 1e-8).
        5. weight_decay (float, optional): Weight decay coefficient (default: 0.01).
        6. amsgrad (boolean, optional): Whether to use the AMSGrad variant of this algorithm
                from the paper "On the Convergence of Adam and Beyond" (default: False).
        '''
        # 2. 将所有超参数打包成一个字典
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        self.betas = betas
        super().__init__(params, defaults)


    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]           # Get the learning rate.

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]  # Get state associated with p.
                grad = p.grad.data     # Get the gradient of loss with respect to p.
                t = state.get("t", 1)  # Get iteration number from the state, or initial value.
                weight_decay = group["weight_decay"]

                '''
                    state中需要包含下面这几个记忆化的参数:
                    m: 一阶矩
                    v: 二阶矩
                '''
                m = state.get("m", 0)  # Get iteration number from the state, or initial value.
                v = state.get("v", 0)

                beta1 = group["betas"][0]
                beta2 = group["betas"][1]
                eps = group["eps"]

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad
                
                alpha = lr * (torch.sqrt(1 - torch.pow(torch.tensor(beta2), t)) / (1 - torch.pow(torch.tensor(beta1), t)))
                
                p.data -= alpha * (m / (torch.sqrt(v) + eps))
                p.data -= lr * weight_decay * p.data

                state["t"] = t + 1     # Increment iteration number.
                state["m"] = m
                state["v"] = v

        return loss
    


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    计算一个带有预热 warm-up 和余弦退火 cosine annealing 的学习率。
    - 阶段1: 预热 (it < warmup_iters)
      学习率从 0 线性增加到 max_learning_rate。
    - 阶段2: 余弦退火 (warmup_iters <= it <= cosine_cycle_iters)
      学习率按照余弦曲线从 max_learning_rate 平滑下降到 min_learning_rate。
    - 阶段3: 保持最小 (it > cosine_cycle_iters)
      学习率固定为 min_learning_rate。
    """
    # 1. 预热阶段
    if it < warmup_iters:
        return float(max_learning_rate * it) / warmup_iters

    # 2. 后退火阶段 (保持最小学习率)
    if it > cosine_cycle_iters:
        return min_learning_rate

    # 3. 余弦退火阶段
    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    
    #    计算余弦衰减系数 (从 1.0 到 0.0)
    #    根据公式，输入 cos 的值需要乘以 pi
    #    torch.cos 的输入必须是 tensor
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    #    根据系数和范围计算最终学习率
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)




def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    # 过滤掉没有梯度的参数
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return

    # 1. 第一次遍历 (隐式): 计算所有梯度的总 L2 范数
    # detach() 在计算总范数时，我们只关心梯度的数值，不需要追踪这个计算过程本身。
    grads = [p.grad.detach() for p in params_with_grad]
    
    total_norm = 0.0
    for grad in grads:
        total_norm += grad.pow(2).sum()
    total_norm = total_norm.sqrt() # total_norm 现在是一个单独的数字

    if total_norm > max_l2_norm:
        # 计算缩放系数
        clip_coef = max_l2_norm / (total_norm + 1e-6)

        for p in params_with_grad:
            # 就地修改所有梯度
            # detach() 安全地进行就地修改：防止在修改梯度值时干扰计算图，避免运行时错误。
            p.grad.detach().mul_(clip_coef)

