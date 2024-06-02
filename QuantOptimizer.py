from torch.optim import Optimizer
import torch

class QuantOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(QuantOptimizer, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
#                 print(p.data.shape)
                p.data = p.data - p.grad * group['lr']
        return 
    
    
    def PowerOfTwo(self, x):
        return 2**(torch.round(torch.log2(x)))