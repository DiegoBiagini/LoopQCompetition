from torch import dropout, nn
import math


class CNNLSTM(nn.Module):

    def __init__(self, n_classes, init_w, init_h, inference=False) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.init_w = init_w
        self.init_h = init_h

        self.lflbs = nn.Sequential(
            LFLB(in_channels=1, out_channels=32, conv_kernel_size=(3,3), conv_stride=(1,1), maxpool_kernel_size=(2,2)),
            LFLB(in_channels=32, out_channels=48, conv_kernel_size=(3,3), conv_stride=(1,1), maxpool_kernel_size=(2,2)),
            LFLB(in_channels=48, out_channels=64, conv_kernel_size=(3,3), conv_stride=(1,1), maxpool_kernel_size=(2,2)),
            LFLB(in_channels=64, out_channels=128, conv_kernel_size=(3,3), conv_stride=(1,1), maxpool_kernel_size=(4,4)),
            LFLB(in_channels=128, out_channels=256, conv_kernel_size=(3,3), conv_stride=(1,1), maxpool_kernel_size=(4,4)),
        )

        lstm_in_size = 256*math.floor(init_h/128.)*math.floor(init_w/128.)
        self.lstm = nn.LSTM(input_size=lstm_in_size, hidden_size=256, num_layers=2, dropout=0.5)

        self.head = nn.Sequential(
            nn.Linear(in_features=256, out_features=n_classes)
        )
        if inference:
            self.head.append(nn.Softmax(dim=1))
    


    def forward(self, input):
        x = self.lflbs(input)
        x = nn.Flatten(start_dim=1)(x)
        #x = nn.Dropout(p=0.5)(x)
        x = self.lstm(x)[0] # Take only the output of the LSTM
        x = nn.Dropout(p=0.5)(x)
        out = self.head(x)
        return out


class LFLB(nn.Module):

    def __init__(self, in_channels, out_channels, conv_kernel_size, conv_stride, maxpool_kernel_size) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                kernel_size=conv_kernel_size, stride=conv_stride, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= maxpool_kernel_size)

        )

    def forward(self, input):
        return self.block(input)