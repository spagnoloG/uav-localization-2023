#!/usr/bin/env python3

# Pytorch: [batch_size, channels, height, width]
# numpy: [height, width, channels]
# [batch_size, height, width, channels] -> [batch_size, channels, height, width]

import torch.nn as nn
import timm
import torch
import torch.nn.functional as F


class FusionModule(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_size):
        super(FusionModule, self).__init__()
        self.upsample_size = upsample_size

        self.conv1 = nn.Conv2d(
            in_channels=in_channels[0], out_channels=out_channels, kernel_size=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels[1], out_channels=out_channels, kernel_size=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=in_channels[2], out_channels=out_channels, kernel_size=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        self.weights = nn.Parameter(torch.ones(3, 1, 1, 1))

    def forward(self, drone1, drone2, drone3, satellite):
        drone1 = self.conv1(drone1)
        drone2 = self.conv2(drone2)
        drone3 = self.conv3(drone3)

        drone2_up = F.interpolate(
            drone2, size=drone1.shape[2:], mode="bilinear", align_corners=False
        )
        drone3_up = F.interpolate(
            drone3, size=drone1.shape[2:], mode="bilinear", align_corners=False
        )

        # Fuse the upsampled feature maps with the feature maps of the same scale
        drone1_fused = drone1 + drone2_up + drone3_up

        # Further extract features using a 3x3 convolution
        drone1_fused = self.conv4(drone1_fused)

        drone1_up = F.interpolate(
            drone1_fused, size=satellite.shape[2:], mode="bilinear", align_corners=False
        )
        drone2_up = F.interpolate(
            drone2_up, size=satellite.shape[2:], mode="bilinear", align_corners=False
        )
        drone3_up = F.interpolate(
            drone3_up, size=satellite.shape[2:], mode="bilinear", align_corners=False
        )

        A1 = F.cosine_similarity(satellite, drone1_up, dim=1).unsqueeze(1)
        A2 = F.cosine_similarity(satellite, drone2_up, dim=1).unsqueeze(1)
        A3 = F.cosine_similarity(satellite, drone3_up, dim=1).unsqueeze(1)

        # Weighted fusion
        fusion = A1 * self.weights[0] + A2 * self.weights[1] + A3 * self.weights[2]

        self.weights.data = self.weights.data / self.weights.data.sum()

        # Sum along the channel dimension to get the final fused feature map
        fusion = fusion.sum(dim=1, keepdim=True)

        fusion = F.interpolate(
            fusion, size=self.upsample_size, mode="bilinear", align_corners=False
        )

        return fusion


class SaveLayerFeatures(nn.Module):
    def __init__(self):
        super(SaveLayerFeatures, self).__init__()
        self.outputs = []

    def forward(self, x, shape):
        output = x.clone()
        output = output.reshape(x.shape[0], x.shape[2], shape[0], shape[1])
        self.outputs.append(output)
        return x

    def clear(self):
        self.outputs = []


class ModifiedPCPVT(nn.Module):
    def __init__(self, original_model):
        super(ModifiedPCPVT, self).__init__()

        # Change the structure of the PCPVT model
        self.model = original_model
        self.model.blocks = original_model.blocks[:3]  # Only use the first 3 blocks
        self.model.norm = nn.Identity()  # Remove the normalization layer
        self.model.head = nn.Identity()  # Remove the head layer
        self.model.patch_embeds[
            3
        ] = nn.Identity()  # Remove the last patch embedding layer
        self.model.pos_block[
            3
        ] = nn.Identity()  # Remove the last position embedding layer

        # Add the save_features layer to the first 3 blocks
        self.save_features = SaveLayerFeatures()
        self.model.blocks[0].add_module("save_features", self.save_features)
        self.model.blocks[1].add_module("save_features", self.save_features)
        self.model.blocks[2].add_module("save_features", self.save_features)

    def forward(self, x):
        _ = self.model(x)
        features = self.save_features.outputs.copy()
        self.save_features.clear()
        return features


class CrossViewLocalizationModel(nn.Module):
    def __init__(self, drone_resolution, satellite_resolution):
        super(CrossViewLocalizationModel, self).__init__()

        self.drone_resolution = drone_resolution
        self.satellite_resolution = satellite_resolution

        # Feature extraction module
        self.backbone_UAV = timm.create_model("twins_pcpvt_small", pretrained=True)
        self.feature_extractor_UAV = ModifiedPCPVT(self.backbone_UAV)

        self.backbone_satellite = timm.create_model(
            "twins_pcpvt_small", pretrained=True
        )

        self.feature_extractor_satellite = ModifiedPCPVT(self.backbone_satellite)

        # Weight-Adaptive Multi-Feature fusion module
        self.fusion = FusionModule(
            in_channels=[64, 128, 320],
            out_channels=64,
            upsample_size=self.satellite_resolution,
        )

        # Upsampling module
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x_UAV, x_satellite):
        # Pytorch: [batch_size, channels, height, width]
        # numpy: [height, width, channels]
        x_UAV = x_UAV.permute(0, 3, 1, 2)
        x_satellite = x_satellite.permute(0, 3, 1, 2)

        feature_pyramid_UAV = self.feature_extractor_UAV(x_UAV)
        feature_pyramid_satellite = self.feature_extractor_satellite(x_satellite)
        last_sat_feature = feature_pyramid_satellite[0]

        fused_map = self.fusion(*feature_pyramid_UAV, last_sat_feature)

        return fused_map
