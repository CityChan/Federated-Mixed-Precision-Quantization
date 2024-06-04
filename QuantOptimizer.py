from torch.optim import Optimizer
import torch
import numpy as np
from models.bit import BitLinear, BitConv2d

class QuantOptimizer(Optimizer):
    def __init__(self, model,params, lr, momentum, weight_decay):
        defaults = dict(lr=lr, momentum = momentum, weight_decay=weight_decay)
        super(QuantOptimizer, self).__init__(params, defaults)
        self.model = model
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if group['momentum'] != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(group['momentum']).add_(d_p, alpha = 1)
                    d_p = buf
                if len(d_p.data.shape) == 3 or  len(d_p.data.shape) == 5:
                    
                    
                    Nbits = d_p.data.shape[-1]
                    dev = d_p.device
                    ex = np.arange(Nbits-1, -1, -1)
                    exps = torch.Tensor(2**ex).float()
                    sign = torch.where(d_p.data > 0, torch.full_like(d_p.data, 1), torch.full_like(d_p.data, -1))
                    dB = torch.mul(self.PowerOfTwo(abs(d_p.data)), exps.to(dev))
                    dB = torch.mul(dB,sign)
                    dB = torch.div(dB, exps.to(dev))
                    for i in range(Nbits):
                        ex = 2**(Nbits-i-1)
                        d_p.data[...,i] = dB[...,i]

                    p.data.add_(d_p.data, alpha = -group['lr'])
                    
                else:
                    p.data.add_(d_p.data, alpha = -group['lr'])
        
        for name, module in self.model.named_modules():
                if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
                    module.quant(maxbit = 8)
                    
        return 
    
    
    def PowerOfTwo(self, x):
        return 2**(torch.round(torch.log2(x)))