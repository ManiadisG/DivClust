import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiheadLinear(nn.Module):
    def __init__(self, d_in, d_out, parallel_no, bias=False, same_weight_init=False):
        super(MultiheadLinear, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.parallel_no = parallel_no
        self.add_bias = bias

        weights = [torch.empty((d_out, d_in)) for k_ in range(parallel_no)]
        if self.add_bias:
            bias = [torch.empty((1, d_out)) for k_ in range(parallel_no)]
        else:
            bias = [None]*parallel_no
        initialized_weights, initialized_bias = [],[]
        for w, b in zip(weights, bias):
            w_init, b_init = self.init_weights(w, b)
            initialized_weights.append(w_init)
            initialized_bias.append(b_init)

        self.weight = nn.Parameter(torch.stack(initialized_weights))
        if same_weight_init:
            for k in range(parallel_no):
                self.weight.data[k] = self.weight.data[0]

        if self.add_bias:
            self.bias = nn.Parameter(torch.stack(initialized_bias))
            if same_weight_init:
                for k in range(parallel_no):
                    self.bias.data[k] = self.bias.data[0]
        else:
            self.bias = None

    def init_weights(self, weight, bias=None):
        from torch.nn.modules.linear import init
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(bias, -bound, bound)
        return weight, bias

    def forward(self, x):
        x_shape = x.shape
        if len(x_shape) == 2:
            r = torch.einsum('ik,bjk->bij', x, self.weight)
        elif len(x_shape) == 3:
            if x_shape[0] == 1:
                x = x.squeeze(0)
                r = torch.einsum('ik,bjk->bij', x, self.weight)
            else:
                r = torch.einsum('bik,bjk->bij', x, self.weight)
        if self.add_bias:
            return r + self.bias
        else:
            return r