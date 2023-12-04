import torch
from torch import Tensor
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.types import Number

class axs(Module):
    def __init__(self, x_in: int, y_in: int, x_out: int, y_out: int, device=None, dtype=None, x_end: Number = 1, y_end: Number = 1) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.x_in = x_in
        self.y_in = y_in
        self.x_out = x_out
        self.y_out = y_out
        self.x_end = x_end
        self.y_end = y_end
        self.out_features = self.x_out * self.y_out
        self.pos2d = Parameter(torch.empty((self.y_out, self.x_out, 2), **factory_kwargs))
        self.weight = Parameter(torch.empty((self.y_out, self.x_out), **factory_kwargs))
        self.weightf = nn.Sequential(
            nn.RReLU(1.1, 1.15),
        )
        self.ep = Parameter(torch.empty((self.y_out, self.x_out), **factory_kwargs))
        self.epf = nn.Sequential(
            nn.LogSigmoid(),
            nn.RReLU(0.013, 0.017),
        )
        self.dis = Parameter(torch.empty((self.y_out, self.x_out), **factory_kwargs))
        self.disf = nn.Sequential(
            nn.RReLU(2., 2.3),
            nn.Sigmoid(),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        np2d = torch.stack(torch.meshgrid(torch.linspace(0, self.x_end, steps=self.x_out), torch.linspace(0, self.y_end, steps=self.y_out), indexing='xy'), dim=-1)
        self.pos2d = Parameter(np2d)
        nw = torch.empty_like(self.weight)
        nw.fill_(0.167)
        self.weight = Parameter(nw)
        ep = torch.empty_like(self.ep)
        ep.fill_(-0.1)
        self.ep = Parameter(ep)
        dis = torch.empty_like(self.dis)
        dis.fill_(-1.5)
        self.dis = Parameter(dis)

    def forward(self, input: Tensor) -> Tensor:
        grid = torch.stack(torch.meshgrid(torch.linspace(0, self.x_end, steps=self.x_in), torch.linspace(0, self.y_end, steps=self.y_in), indexing='xy'), dim=-1).unsqueeze(-2).unsqueeze(-2).repeat(1,1,self.y_out,self.x_out,1).to(self.pos2d.device)
        exp_pos = grid - self.pos2d
        exp_dis_square = torch.sum(exp_pos**2, dim=-1)
        exp = torch.where(exp_dis_square < self.disf(self.dis), torch.exp(self.epf(self.ep) * exp_dis_square), False) # 俺寻思可以去掉按距离筛选
        weight = exp.permute(2,3,0,1).reshape(self.out_features,-1)
        i2d = input.reshape(input.size(0), -1).permute(1, 0)
        o21 = torch.matmul(weight, i2d)
        output = o21.permute(1, 0).reshape(input.size(0), input.size(1), self.y_out, self.x_out) * self.weightf(self.weight)
        return output
