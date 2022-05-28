from torch import nn
from transformers import HubertModel, PretrainedConfig, AutoConfig
import torch

class HUBERTFT(nn.Module):

    def __init__(self, n_classes, inference = False, load_weights = True) -> None:
        super().__init__()
        self.n_classes = n_classes
        if load_weights:
            self.hubert = HubertModel.from_pretrained("facebook/hubert-large-ll60k")
        else:
            self.hubert = HubertModel(config = AutoConfig.from_pretrained("facebook/hubert-large-ll60k"))
        self.head = nn.Sequential(
            nn.Linear(in_features=1024, out_features=n_classes)
        )
        if inference:
            self.head.append(nn.Softmax(-1))

    def forward(self, input, attention_mask = None):
        if attention_mask is None:
            x = self.hubert(torch.squeeze(input)).last_hidden_state
        else:
            x = self.hubert(torch.squeeze(input), attention_mask = torch.squeeze(attention_mask)).last_hidden_state

        x = torch.transpose(x, 1, 2)
        x = torch.squeeze(nn.AdaptiveAvgPool1d(1)(x))
        
        out = self.head(x)

        return out

