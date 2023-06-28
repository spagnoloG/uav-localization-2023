#!/usr/bin/env python3

# Pytorch: [batch_size, channels, height, width]
# numpy: [height, width, channels]
# [batch_size, height, width, channels] -> [batch_size, channels, height, width]

import torch.nn as nn
import timm
import torch
import torch.nn.functional as F


class FusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionModule, self).__init__()
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

        # Upsample drone features to match the size of the satellite features
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

        # Weighted fusion
        fusion = (
            A1 * self.weights[0]
            + drone2_up * self.weights[1]
            + drone3_up * self.weights[2]
        )

        self.weights.data = self.weights.data / self.weights.data.sum()

        # Sum along the channel dimension to get the final fused feature map
        fusion = fusion.sum(dim=1, keepdim=True)

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
    def __init__(self):
        super(CrossViewLocalizationModel, self).__init__()

        # Feature extraction module
        self.backbone_UAV = timm.create_model("twins_pcpvt_small", pretrained=True)
        self.feature_extractor_UAV = ModifiedPCPVT(self.backbone_UAV)

        self.backbone_satellite = timm.create_model(
            "twins_pcpvt_small", pretrained=True
        )

        self.feature_extractor_satellite = ModifiedPCPVT(self.backbone_satellite)

        # Weight-Adaptive Multi-Feature fusion module
        self.fusion = FusionModule(in_channels=[64, 128, 320], out_channels=64)

        # Upsampling module
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x_UAV, x_satellite):
        # Pytorch: [batch_size, channels, height, width]
        # numpy: [height, width, channels]
        print("Shape of x_UAV before permute: ", x_UAV.shape)
        print("Shape of x_satellite before permute: ", x_satellite.shape)
        x_UAV = x_UAV.permute(0, 3, 1, 2)
        x_satellite = x_satellite.permute(0, 3, 1, 2)
        print("Shape of x_UAV after permute: ", x_UAV.shape)
        print("Shape of x_satellite after permute: ", x_satellite.shape)
        feature_pyramid_UAV = self.feature_extractor_UAV(x_UAV)
        feature_pyramid_satellite = self.feature_extractor_satellite(x_satellite)

        print("Printing the shape of feature_pyramid_UAV: ")
        for feature in feature_pyramid_UAV:
            print(feature.shape)

        print("Printing the shape of feature_pyramid_satellite: ")
        for feature in feature_pyramid_satellite:
            print(feature.shape)

        last_sat_feature = feature_pyramid_satellite[-1]

        exit()

        # Calculate similarity and perform weighted fusion
        heatmaps = []
        for uav_feature in feature_pyramid_UAV:
            heatmap = self.wamf(uav_feature, last_sat_feature)
            heatmaps.append(heatmap)

        print("Printing the shape of heatmaps: ")

        # Upsample heatmaps to the same size
        heatmaps = [self.upsample(heatmap) for heatmap in heatmaps]

        # Sum the heatmaps to get the final heatmap
        final_heatmap = sum(heatmaps)

        exit()


def test():
    fusion_module = FusionModule(in_channels=[64, 128, 320], out_channels=64)
    satellite = torch.randn(2, 64, 100, 100)  # S3
    drone1 = torch.randn(2, 64, 32, 32)  # U1
    drone2 = torch.randn(2, 128, 16, 16)  # U2
    drone3 = torch.randn(2, 320, 8, 8)  # U3
    fusion = fusion_module(drone1, drone2, drone3, satellite)
    print(fusion.shape)


if __name__ == "__main__":
    test()
