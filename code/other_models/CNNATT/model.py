from torch import nn
from typing import List, Tuple
import math
import torch

class CNNATT(nn.Module):

    def __init__(self, n_classes, inference=False) -> None:
        super().__init__()
        self.n_classes = n_classes

        self.split_path = SplitPath(1, 8, [(11,3),(3,9)])

        self.convolutions = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2,2)),

            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(3,3), padding="same"),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2,2)),

            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3,3), padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=80, kernel_size=(3,3), padding="same"),
            nn.BatchNorm2d(80),
            nn.ReLU()
        )
        
        self.attention = DotSelfAttention(in_size=80, out_size=128, n_heads=3)
        self.ln = nn.LayerNorm(80)

        self.head = nn.Sequential(
 
            nn.Linear(128,n_classes)
            )
        if inference:
            self.head.append(nn.Softmax())

    def forward(self, input):
        
        x = self.split_path(input)
        x = self.convolutions(x)

        x = nn.Flatten(start_dim=2)(x)
        #x = nn.Dropout(0.3)(x)
        x = self.attention(x)

        x = nn.AdaptiveAvgPool1d(1)(x)
        x = torch.squeeze(x)
        x = self.ln(x)

        out = self.head(x)
        return out

class SplitPath(nn.Module):

    def __init__(self, in_channels, out_channels, kernels : List[Tuple[int,int]]) -> None:
        super().__init__()

        self.paths = nn.ModuleList()
        for k in kernels:
            self.paths.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k, padding="same"),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )
            )

    def forward(self, input):
        outs = []
        for path in self.paths:
            outs += [path(input)]
        return torch.concat(outs, dim=1)

class DotSelfAttention(nn.Module):

    def __init__(self, in_size, out_size, n_heads) -> None:
        super().__init__()

        self.Qs = nn.ParameterList()
        self.Ks = nn.ParameterList()
        self.Vs = nn.ParameterList()

        self.norm = math.sqrt(in_size)
        self.in_size = in_size
        self.out_size = out_size
        self.n_heads = n_heads

        init_fn = nn.init.xavier_uniform_
        for i in range(n_heads):
            self.Qs.append(nn.parameter.Parameter(init_fn(torch.zeros(size=(out_size, in_size)))))
            self.Ks.append(nn.parameter.Parameter(init_fn(torch.zeros(size=(out_size, in_size)))))
            self.Vs.append(nn.parameter.Parameter(init_fn(torch.zeros(size=(out_size, in_size)))))


    def forward(self, input):
        batch_size = input.shape[0]

        Xatts = None

        for i in range(self.n_heads):

            K = torch.bmm(self.Ks[i].repeat(batch_size, 1, 1), input)
            Q = torch.bmm(self.Qs[i].repeat(batch_size, 1, 1), input)
            V = torch.bmm(self.Vs[i].repeat(batch_size, 1, 1), input)

            x = nn.Softmax(dim=-1)(torch.div(torch.bmm(K, torch.transpose(Q, 1, 2)), self.norm))
            x = torch.bmm(x, V)

            if Xatts is None:
                Xatts = x
            else:
                Xatts = torch.add(Xatts, x)

        return torch.div(Xatts, self.n_heads)