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
                    exps = torch.Tensor(2**ex).float().to(dev)

                    sign = torch.where(d_p.data > 0, torch.full_like(d_p.data, 1), torch.full_like(d_p.data, -1))
                    grad_scale = torch.max(abs(d_p.data))/torch.tensor(2**Nbits-1).to(dev)
                    scaled_dp = torch.div(abs(d_p.data), grad_scale)

                    dW = torch.mul(self.PowerOfTwo(scaled_dp*group['lr']), exps.to(dev))
                    dB = torch.log2(dW)
                    valid_mask = (dB >= 0) & (dB <= Nbits)
                    power_positive = torch.zeros_like(dB)
                    power_positive[valid_mask] = 2 ** dB[valid_mask]

                    negative_mask = dB < 0
                    power_negative = torch.zeros_like(dB)
                    power_negative[negative_mask] = 2 ** dB[negative_mask]
                    p_bern = torch.sum(power_negative, dim = -1)
                    dist = torch.distributions.Bernoulli(p_bern.clamp(min=0.0, max=1.0))
                    dist.sample()

                    dB = torch.mul(torch.mul(power_positive, grad_scale), sign)
                    db_bern = torch.mul(torch.mul(dist.sample(), grad_scale), sign[..., 0])
                    for i in range(Nbits):
                        d_p.data[...,i]  = dB[...,i]
                    d_p.data[...,-1] = d_p.data[...,-1] + db_bern
                    p.data.add_(d_p.data, alpha = -group['lr'])
                    
                else:
                    p.data.add_(d_p.data, alpha = -group['lr'])
        
        for name, module in self.model.named_modules():
                if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
                    module.quant(maxbit = 8)
                    
        return 

    
    def PowerOfTwo(self, x):
        return 2**(torch.round(torch.log2(x)))