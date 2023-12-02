import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

class axs(Module):
    def __init__(self, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.pos2d = Parameter(torch.empty((28, 28, 2), **factory_kwargs))
        self.weight = Parameter(torch.empty((28, 28), **factory_kwargs))
        self.reset_parameters()

    def weight_by_distance(self, x):
        # 此处的-0.5可以改为可调参数以变换函数曲线
        exponent = -0.5 * torch.sum(x**2)
        return torch.exp(exponent)
    
    def get_value(self, x, pos):
        if pos[0] < x.size(0) and pos[1] < x.size(1) and pos[0] >= 0 and pos[1] >= 0:
            return x[pos[0], pos[1]]
        else:
            return torch.tensor(0.)
        
    def calculate_single_value(self, x, pos, center):
        v = self.get_value(x, pos)
        if torch.allclose(v, torch.tensor(0.)):
            return v
        else:
            dis = pos - center
            w = self.weight_by_distance(dis)
            return v * w
        
    def sum_single_value_with_weight(self, x, pos, weight):
        ce = torch.round(pos)
        v = torch.empty((5, 5), device=pos.device)
        for i0 in range(-2, 3):
            for i1 in range(-2, 3):
                cb = torch.tensor([i0, i1], device=pos.device)
                cr = ce + cb
                v[i0, i1] = self.calculate_single_value(x, cr.int(), pos)
        vw = v * weight
        return vw.sum()
        
    def reset_parameters(self) -> None:
        np2d = torch.empty_like(self.pos2d)
        for d0 in range(self.pos2d.size(0)):
            for d1 in range(self.pos2d.size(1)):
                np2d[d0, d1] = torch.tensor([d0, d1])
        self.pos2d = Parameter(np2d.float())
        nw = torch.empty_like(self.weight)
        nw.fill_(0.167)
        self.weight = Parameter(nw)

    def forward(self, input: Tensor) -> Tensor:
        output = torch.empty_like(input, device=input.device)
        if input.dim() == 4:
            for bn in range(input.size(0)):
                sw = F.relu(self.weight)
                for d0 in range(28):
                    for d1 in range(28):
                        # 此处可以增加通过输出与输入中心坐标距离控制权重取值的函数(如1-ReLU)，以图在输入中心偏离过多时停止的梯度传递，以此限制输入中心的偏移量。
                        output[bn, 0, d0, d1] = self.sum_single_value_with_weight(input[bn, 0], self.pos2d[d0, d1], sw[d0, d1])
        elif input.dim() == 3:
            for bn in range(input.size(0)):
                sw = F.relu(self.weight)
                for d0 in range(28):
                    for d1 in range(28):
                        # 此处可以增加通过输出与输入中心坐标距离控制权重取值的函数(如1-ReLU)，以图在输入中心偏离过多时停止的梯度传递，以此限制输入中心的偏移量。
                        output[bn, d0, d1] = self.sum_single_value_with_weight(input[bn], self.pos2d[d0, d1], sw[d0, d1])
        elif input.dim() == 2:
            for bn in range(input.size(0)):
                sw = F.relu(self.weight)
                for d0 in range(28):
                    for d1 in range(28):
                        # 此处可以增加通过输出与输入中心坐标距离控制权重取值的函数(如1-ReLU)，以图在输入中心偏离过多时停止的梯度传递，以此限制输入中心的偏移量。
                        output[d0, d1] = self.sum_single_value_with_weight(input, self.pos2d[d0, d1], sw[d0, d1])
        return output

    def extra_repr(self) -> str:
        return 'd0={}, d1={}, bias={}'.format(
            self.pos2d.size(0), self.pos2d.size(1), self.bias is not None
        )