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
        self.ep = Parameter(torch.empty((28, 28), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        np2d = torch.stack(torch.meshgrid(torch.arange(0, 28), torch.arange(0, 28), indexing='xy'), dim=-1)
        self.pos2d = Parameter(np2d.float())
        nw = torch.empty_like(self.weight)
        nw.fill_(0.167)
        self.weight = Parameter(nw)
        ep = torch.empty_like(self.ep)
        ep.fill_(-0.1)
        self.weight = Parameter(ep)

    def forward(self, input: Tensor) -> Tensor:
        grid = torch.stack(torch.meshgrid(torch.arange(0, 28), torch.arange(0, 28), indexing='xy'), dim=-1).unsqueeze(-2).unsqueeze(-2).repeat(1,1,28,28,1).to(self.pos2d.device)
        exp_pos = grid - self.pos2d
        exp_dis_square = torch.sum(exp_pos**2, dim=-1)
        exp_vaild = torch.where(exp_dis_square < 9, True, False) # 俺寻思这里的9可以改成可调参数或可学习参数，或者可以去掉exp_vaild
        ep = -F.relu(-self.ep)
        exp = torch.exp(ep * exp_dis_square) # 俺寻思这里的ep可以改成固定参数或手动设置参数
        exp = exp * exp_vaild
        weight = exp.reshape(784,-1)
        i2d = input.reshape(input.size(0), -1).permute(1, 0)
        o21 = torch.matmul(weight, i2d)
        output = o21.permute(1, 0).reshape(input.size()) * self.weight
        return output

    def extra_repr(self) -> str:
        return 'd0={}, d1={}, bias={}'.format(
            self.pos2d.size(0), self.pos2d.size(1), self.bias is not None
        )