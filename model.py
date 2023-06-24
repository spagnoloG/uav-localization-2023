#!/usr/bin/env python3
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


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


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 384):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
            nn.Transpose(1, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        return x


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
        self.conv = nn.Conv2d(
            emb_size, 1, kernel_size=1
        )  # 1x1 convolution to merge the features

    def forward(self, drone_img, satellite_img):
        drone_features = self.resnet50(drone_img)
        satellite_features = self.resnet50(satellite_img)

        drone_features = self.positional_encoding(drone_features)
        satellite_features = self.positional_encoding(satellite_features)

        drone_transformer_features = self.transformer(drone_features)
        satellite_transformer_features = self.transformer(satellite_features)

        # concatenate the features along the channel dimension
        concat_features = torch.cat(
            (drone_transformer_features, satellite_transformer_features), dim=1
        )

        # pass the concatenated features through the 1x1 conv layer to generate the heatmap
        heatmap = self.conv(concat_features)

        return heatmap


class BalanceLoss(nn.Module):
    def __init__(self, w_neg=1.0, R=1):
        super(BalanceLoss, self).__init__()
        self.w_neg = w_neg
        self.R = R

    def forward(self, heatmap, label):
        # Step 1: generate the 0,1 matrix
        t = (label >= self.R).float()

        # Step 2: copy t to w
        w = t.clone()

        # Step 3 and 4: num of the positive and negative samples
        N_pos = self.R**2
        N_neg = heatmap.numel() - N_pos

        # Step 5 and 6: weight of the positive and negative samples
        W_pos = 1.0 / N_pos
        W_neg = (1.0 / N_neg) * self.w_neg

        # Assign weights to w
        w[t == 1] = W_pos
        w[t == 0] = W_neg

        # Step 7: weight normalization
        w = w / torch.sum(w)

        # Step 8: map normalization
        p = torch.sigmoid(heatmap)

        # Step 9: balance loss
        loss = -torch.sum((t * torch.log(p) + (1 - t) * torch.log(1 - p)) * w)

        return loss
