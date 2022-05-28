from typing import List, Tuple
import torch
from torch import nn


class FC_MFCC(nn.Module):
    """Model that works on the MFCC of an audio file
    It is made up of 3 parts:
    - Initial multi-path convolution (3 towers) to consider time and frequency correlations
    - Middle fully convolutional feature learning alternating convolutions, BN, avgPool
    - Classification head

    """


    def __init__(self, n_classes, inference = False) -> None:
        super().__init__()

        self.n_classes = n_classes

        self.initial = MP_Conv(
            in_channels=1, conv_channels=32,
            conv_sizes= [(3,3), (9,1), (1,11)], avg_pool_size=(2,2)
        )
        self.middle = Feature_Learning(
            in_channels = 32*3, n_structures=5, kernel_size=(3,3),
            conv_channel_sizes=(64, 96, 128, 160, 320),
            avg_pool_sizes=[(2,2),(2,2), (2,1), (2,1)]
        )
        self.head = torch.nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(320, n_classes)
        )
        if inference:
            self.head.append(nn.Softmax(-1))
    
    def forward(self, input) -> torch.Tensor:
        x = self.initial(input)
        x = self.middle(x)
        x = torch.squeeze(x)
        out = self.head(x)
        return out


class MP_Conv(nn.Module):
    """Part of the model which generates feature maps from the input MFCCs
    Creates N paths, each path is composed of: Convolution, BatchNorm, ReLU, AvgPool
    Then concatenate the resulting feature maps along the channels
    """

    def __init__(self,in_channels : int, conv_channels : int, conv_sizes : List[Tuple[int,int]], avg_pool_size : Tuple[int,int]) -> None:
        super().__init__()
        self.towers = nn.ModuleList()
    
        for size in conv_sizes:
            path = nn.Sequential(
                nn.Conv2d(in_channels, conv_channels, kernel_size=size, padding="same"),
                nn.BatchNorm2d(conv_channels),
                nn.ReLU(),
                nn.ZeroPad2d((0,avg_pool_size[0]-1,0,avg_pool_size[0]-1)),
                nn.AvgPool2d(avg_pool_size)
            )
            self.towers.append(path)

    def forward(self, input) -> torch.Tensor:
        out = None
        for tower in self.towers:
            x = tower(input)
            if out is None:
                out = x
            else:
                out = torch.concat([out, x], dim=1)
        return out

class Feature_Learning(nn.Module):
    """General feature learning part, a succession of N sequential structures
    Each structure is made up of: Conv2d, BatchNorm, Relu, AvgPool
    Each conv has the same kernel size (except the last one which is (1,1)) and different out_channels
    The last uses GlobalAvgPool

    """
    def __init__(self, in_channels : int, n_structures : int, kernel_size : Tuple[int,int], 
                    conv_channel_sizes : List[int], avg_pool_sizes : List[Tuple[int,int]]) -> None:
        super().__init__()

        assert n_structures == len(conv_channel_sizes), "Wrong number of Conv Layers channels"
        assert n_structures - 1 == len(avg_pool_sizes), "Wrong number of AvgPool kernel sizes"

        self.structures = nn.ModuleList()
        for i in range(n_structures):
            in_channels = conv_channel_sizes[i-1] if i != 0 else in_channels

            extra_padding = nn.ZeroPad2d((0, avg_pool_sizes[i][0]- 1, 0, avg_pool_sizes[i][1])) if i != n_structures - 1 else nn.Identity()
            out_pool = nn.AvgPool2d(avg_pool_sizes[i]) if i != n_structures - 1 else nn.AdaptiveAvgPool2d(1)
            
            cur_kernel_size = kernel_size if i != n_structures - 1 else (1,1)

            self.structures.append(nn.Sequential(
                nn.Conv2d(in_channels, conv_channel_sizes[i], cur_kernel_size, padding="same"),
                nn.BatchNorm2d(conv_channel_sizes[i]),
                nn.ReLU(),
                #extra_padding,
                out_pool
            ))

    def forward(self, input):
        x = input
        for structure in self.structures:
            x = structure(x)
        return x