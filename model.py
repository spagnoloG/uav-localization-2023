#!/usr/bin/env python3

# Pytorch: [batch_size, channels, height, width]
# numpy: [height, width, channels]
# [batch_size, height, width, channels] -> [batch_size, channels, height, width]

import torch.nn as nn
import timm
import torch
import torch.nn.functional as F


class WAMF(nn.Module):
    def __init__(self, in_channels):
        super(WAMF, self).__init__()
        self.weight = nn.Parameter(torch.ones(3, 1, 1, 1))
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x_UAV, x_satellite):
        # Compute similarity
        similarity_maps = []
        for feature_UAV, feature_satellite in zip(x_UAV, x_satellite):
            print("Shape of feature_UAV: ", feature_UAV.shape)
            print("Shape of feature_satellite: ", feature_satellite.shape)
            feature_UAV = feature_UAV.unsqueeze(0)
            feature_satellite = feature_satellite.unsqueeze(0)
            print("Shape of feature_UAV after unsqueeze: ", feature_UAV.shape)
            print("Shape of feature_satellite after unsqueeze: ", feature_satellite.shape)
            feature_UAV = F.interpolate(feature_UAV, size=feature_satellite.shape[2:], mode='bilinear', align_corners=False)
            similarity_map = self.conv(feature_UAV * feature_satellite)
            similarity_maps.append(similarity_map)

        # Weighted fusion
        fusion = sum(w * heatmap for w, heatmap in zip(self.weight, similarity_maps))
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
        self.wamf = WAMF(64)

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
