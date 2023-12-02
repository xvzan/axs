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
        
    def sum_single_value_with_weight(self, x, pos, weight):
        ce = torch.round(pos)
        exp_center = ce - pos
        start_pos = (ce + 3).clamp_(min=0, max=28).int()
        exp_pos = torch.stack(torch.meshgrid(torch.arange(-2, 3), torch.arange(-2, 3), indexing='xy'), dim=-1).to(pos.device) + exp_center
        exp_weights = torch.exp(-0.5 * torch.sum(exp_pos**2, dim=2))
        x = F.pad(x, (5,5,5,5))
        v_source = x[start_pos[0] : start_pos[0]+5, start_pos[1] : start_pos[1]+5]
        v = v_source * exp_weights
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
                        output[bn, d0, d1] = self.sum_single_value_with_weight(input[bn], self.pos2d[d0, d1], sw[d0, d1])
        elif input.dim() == 2:
            for bn in range(input.size(0)):
                sw = F.relu(self.weight)
                for d0 in range(28):
                    for d1 in range(28):
                        output[d0, d1] = self.sum_single_value_with_weight(input, self.pos2d[d0, d1], sw[d0, d1])
        return output

    def extra_repr(self) -> str:
        return 'd0={}, d1={}, bias={}'.format(
            self.pos2d.size(0), self.pos2d.size(1), self.bias is not None
        )