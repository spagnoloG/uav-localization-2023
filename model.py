#!/usr/bin/env python3
import torch
import torch.nn as nn
import torchvision.models as models


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_size, 2).float()
            * -(torch.log(torch.tensor(10000.0)) / emb_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ImageTransformer(nn.Module):
    def __init__(self, emb_size=384, nhead=4, num_layers=12, dropout=0.5):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size, nhead=nhead, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )

    def forward(self, x):
        x = self.transformer_encoder(x)
        return x


class CustomResNetDeiT(nn.Module):
    def __init__(self, nhead=4, num_layers=12):
        super().__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50 = nn.Sequential(
            *list(self.resnet50.children())[:-1]
        )  # Remove the last FC layer

        emb_size = self.resnet50.fc.in_features
        self.positional_encoding = PositionalEncoding(emb_size)
        self.transformer = ImageTransformer(emb_size, nhead, num_layers)

    def forward(self, x):
        x = self.resnet50(x)
        x = x.view(x.size(0), x.size(1), -1).transpose(1, 2)  # [B, H'*W', emb_size]
        x = self.positional_encoding(x)
        x = self.transformer(x)

        return x
