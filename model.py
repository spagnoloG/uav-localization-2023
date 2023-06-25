#!/usr/bin/env python3

# Pytorch: [batch_size, channels, height, width]
# numpy: [height, width, channels]
# [batch_size, height, width, channels] -> [batch_size, channels, height, width]

import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from logger import logger


class CustomResNetDeiT(nn.Module):
    def __init__(self):
        super().__init__()

        deit_model = timm.create_model("deit_small_patch16_224", pretrained=True)

        self.deit_backbone = nn.Sequential(*list(deit_model.children())[:-2])

        emb_size = deit_model.embed_dim

        self.conv = nn.Conv2d(2 * emb_size, 1, kernel_size=1)

    def forward(self, drone_img, satellite_img):
        drone_img = drone_img.permute(0, 3, 1, 2)
        satellite_img = satellite_img.permute(0, 3, 1, 2)

        drone_features = self.deit_backbone(drone_img)
        satellite_features = self.deit_backbone(satellite_img)

        # logger.info(f"Drone features shape after backbone: {drone_features.shape}")
        # logger.info(f"Sattelite features shape after backbone: {satellite_features.shape}")

        drone_features = drone_features.flatten(start_dim=2).permute(0, 2, 1)
        satellite_features = satellite_features.flatten(start_dim=2).permute(0, 2, 1)
        # logger.info(f"Drone features shape after flatten: {drone_features.shape}")
        # logger.info(f"Sattelite features shape after flatten: {satellite_features.shape}")

        fused_features = torch.cat((drone_features, satellite_features), dim=1)

        # logger.info(f"Concat features shape: {fused_features.shape}")
        # Make sure that the tensor is 4D: [batch_size, channels, height, width]
        fused_features = fused_features.unsqueeze(-1)
        # logger.info(f"Concat features shape after unsqueeze: {fused_features.shape}")
        heatmap = self.conv(fused_features)

        # logger.info(f"Heatmap shape: {heatmap.shape}")

        return heatmap.squeeze()
