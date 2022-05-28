from torch import nn
from torchvision.models import convnext_base
from transformers import HubertModel,HubertConfig
import torch
from typing import List
import numpy as np

class MMFUSION(nn.Module):
    """
    Main class representing the MMFUSION model
    Made up of 3 modalities:
    -waveform, processed with HuBERT base 
    -spectrogram, processed with ConvNeXt
    -mfcc, processed with a BiLSTM

    The modalities are then fused with the UA block
    """
    def __init__(self, n_classes, inference = False, load_pretrained=True) -> None:
        super().__init__()
        self.n_classes = n_classes

        if load_pretrained:
            self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        else:
            self.hubert = HubertModel(config = HubertConfig(name_or_path="facebook/hubert-base-ls960"))
        hubert_out_seq_length = 249

        # Remove the head from the pretrained cnn
        self.pretrained_cnn = convnext_base(pretrained=load_pretrained)
        convnext_out_length = 36
        # Replase the pooling layer with a flattening one to extract features
        self.pretrained_cnn.avgpool = nn.Flatten(start_dim=2) 
        self.pretrained_cnn.classifier = nn.Identity()
        

        lstm_out_length = 313
        self.lstm = nn.LSTM(input_size = 40, hidden_size = 256, num_layers=2, bidirectional=True, dropout=0.5, batch_first=True)
        
        self.mm_attention = UnifiedAttention(embed_dimension=48*3, n_modes=3, 
            # CNN out: 1024x36
            # LSTM out: 256*2 x (313)
            # Hubert out:  768 x (249)
            input_dimensions=[(1024,convnext_out_length),(256*2,lstm_out_length),(768,hubert_out_seq_length)]
            # MM out : 128 x 598
            )

        self.stacked_attention = UnifiedAttention(embed_dimension=48*3, n_modes=1,
            input_dimensions=[(48*3,598)], input_transform=False
        )

        self.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=48*3, out_features=128),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=n_classes)
        )

        if inference:
            self.head.append(nn.Softmax(-1))

    def forward(self, in_wave, in_mfcc, in_spectro):
        # First path using the spectrogram
        cnn_out = self.pretrained_cnn(in_spectro)

        # Second path using the mfcc
        lstm_out, _ = self.lstm(torch.transpose(torch.squeeze(in_mfcc, 1), 1,2))
        lstm_out = torch.transpose(lstm_out, 1, 2)

        # Third path using the raw waveform
        hubert_out = self.hubert(torch.squeeze(in_wave, 1)).last_hidden_state
        hubert_out = torch.transpose(hubert_out, 1, 2)

        att_out = self.stacked_attention(self.mm_attention(cnn_out, lstm_out, hubert_out))
        att_out = torch.squeeze(nn.AdaptiveAvgPool1d(1)(att_out), 2)
        out = self.head(att_out)

        return out


class GatedSelfAttention(nn.Module):
    """
    Multi head gated self attention
    """

    def __init__(self, in_size, out_size, proj_size, n_heads = 1) -> None:
        super().__init__()

        self.n_heads = n_heads
        # m is the sequence length
        self.in_size = in_size # dx
        self.out_size = out_size # d
        self.head_size = out_size//n_heads # d when using multiple heads
        self.proj_size = proj_size # dg

        init_fn = nn.init.xavier_uniform_

        # Obtain Q, K, V from input
        self.W_qs = nn.ParameterList()
        self.W_ks = nn.ParameterList()
        self.W_vs = nn.ParameterList()
        for i in range(n_heads):
            self.W_qs.append(nn.parameter.Parameter(init_fn(torch.zeros(size=(in_size, self.head_size)))))
            self.W_ks.append(nn.parameter.Parameter(init_fn(torch.zeros(size=(in_size, self.head_size)))))
            self.W_vs.append(nn.parameter.Parameter(init_fn(torch.zeros(size=(in_size, self.head_size)))))

        self.norm = np.sqrt(in_size)

        # Gated dot product
        self.W_gqs = nn.ParameterList()
        self.W_gks = nn.ParameterList()
        self.W_gs = nn.ParameterList()
        for i in range(n_heads):
            self.W_gqs.append(nn.parameter.Parameter(init_fn(torch.zeros(size=(self.head_size, proj_size)))))
            self.W_gks.append(nn.parameter.Parameter(init_fn(torch.zeros(size=(self.head_size, proj_size)))))

            self.W_gs.append(nn.parameter.Parameter(init_fn(torch.zeros(size=(proj_size, 2)))))
        
        # Linear transformation for multi head:
        if n_heads > 1:
            self.W_O = nn.parameter.Parameter(init_fn(torch.zeros(size=(self.out_size,self.out_size))))


    def forward(self, input):
        # Input of shape: B x dx x m
        input = torch.transpose(input, 1, 2) # B x m x dx
        out_concat = []
        for i in range(self.n_heads):
            # Q,K,V of shape B x m x d
            Q = torch.matmul(input, self.W_qs[i])
            K = torch.matmul(input, self.W_ks[i])
            V = torch.matmul(input, self.W_vs[i])

            # Compute the gated dot product
            # GQ,GK of shape: B x m x dg
            GQ = torch.matmul(Q, self.W_gqs[i])
            GK = torch.matmul(K, self.W_gks[i])

            # M of shape: B x m x 2
            M = torch.sigmoid(torch.matmul(torch.mul(GQ,GK), self.W_gs[i]))

            # Extract M_q, M_k and repeat them d times to obtain M_q_tilde, M_k_tilde
            # M_q_tilde, M_k_tilde of shape: B x m x d
            M_q_tilde = torch.unsqueeze(M[:, :, 0],-1).repeat(1, 1, self.head_size)
            M_k_tilde = torch.unsqueeze(M[:, :, 1],-1).repeat(1, 1, self.head_size)

            # Compute attention map A and return AV
            A = torch.softmax(torch.div(
                torch.bmm(
                    torch.mul(Q, M_q_tilde),
                    torch.transpose(torch.mul(K, M_k_tilde), 1,2)
                ), self.norm), -1)
            

            out = torch.bmm(A, V)
            out_concat += [out]

        # heads: B x m x d*heads
        heads = torch.concat(out_concat, dim=2)

        if self.n_heads > 1:
            # Apply a linear transformation
            heads = torch.matmul(heads, self.W_O)

        return torch.transpose(heads, 1, 2)


class UnifiedAttention(nn.Module):
    """
    Unified attention block
    Take into the modalities, apply an optional linear transformation to them, concatenate them and pass them through:
    -GSA with skip connection and LayerNorm
    -Fully connected network with skip connection and LayerNorm
    """
    def __init__(self, embed_dimension, n_modes, input_dimensions : List[int], input_transform=True) -> None:
        super().__init__()
        self.n_modes = n_modes

        self.input_transform = input_transform

        assert n_modes == len(input_dimensions), "Input sizes should be given for all dimensions"
        init_fn = nn.init.xavier_uniform_

        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=embed_dimension, out_features=embed_dimension*4),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=embed_dimension*4, out_features=embed_dimension)
        )

        # Input transformations to reshape the modalities to be of the same channel dimension
        if input_transform:
            self.embed_mappings = nn.ParameterList()
            for i in range(n_modes):
                self.embed_mappings.append(
                    nn.parameter.Parameter(init_fn(torch.zeros(size=(input_dimensions[i][0], embed_dimension)))))
        
        # Layer normalizations along the channel dimension
        concat_seq_len = np.sum([dim[1] for dim in input_dimensions])
        self.ln1 = nn.LayerNorm([embed_dimension, concat_seq_len])
        self.ln2 = nn.LayerNorm([embed_dimension, concat_seq_len])

        # Gated self attention
        self.gsa = GatedSelfAttention(in_size=embed_dimension, out_size=embed_dimension, proj_size=48, n_heads=3)

    def forward(self, *args):
        # Compute the initial transformation
        transform_inputs = []

        for i, input in enumerate(args):
            if self.input_transform:
                transform_inputs += [torch.matmul(torch.transpose(input,1,2), self.embed_mappings[i])]
            else:
                transform_inputs += [torch.transpose(input,1,2)]
        
        concat_inputs = torch.transpose(torch.concat(transform_inputs, dim=1),1,2)

        # Apply gated self attention
        out_gsa = self.gsa(concat_inputs)

        out_gsa = torch.add(out_gsa, concat_inputs)
        out_gsa = self.ln1(out_gsa)

        # Apply fully connected part
        out_fc = torch.transpose(self.fully_connected(torch.transpose(out_gsa,1,2)),1,2)
        out_fc = torch.add(out_fc, out_gsa)
        out_fc = self.ln2(out_fc)

        return out_fc