#!/usr/bin/env python3

# Pytorch: [batch_size, channels, height, width]
# numpy: [height, width, channels]
# [batch_size, height, width, channels] -> [batch_size, channels, height, width]

import torch.nn as nn
import timm
import torch
import torch.nn.functional as F


class MatchCorrelation(nn.Module):
    """Matches the two embeddings using the correlation layer."""

    def __init__(self):
        super(MatchCorrelation, self).__init__()
        self.correlation = F.conv2d

    def forward(self, embed_ref, embed_srch):
        """
        Args:
            embed_ref: (torch.Tensor) The embedding of the reference image. (sat)
            embed_srch: (torch.Tensor) The embedding of the search image. (drone)
        Returns:
            match_map: (torch.Tensor) The correlation map.
        """
        match_map = self.correlation(embed_ref, embed_srch)

        return match_map


class FusionModule(nn.Module):
    """
    Fusion module which applies three convolutional transformations to three drone
    image inputs and combines them through a weighted sum operation.

    Args:
        in_channels: (List[int]) The number of input channels for the three conv layers.
        out_channels: (int) The number of output channels for the three conv layers.
        upsample_size: (Tuple[int, int]) The size to upsample the fused feature map to.

    """

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
        self.weights = nn.Parameter(torch.ones(3, 1, 1, 1))
        self.w_softmax = nn.Softmax(dim=0)

        self.match_correlation = MatchCorrelation()

    def forward(self, drone1, drone2, drone3, satellite):
        drone1 = self.conv1(drone1)
        drone2 = self.conv2(drone2)
        drone3 = self.conv3(drone3)

        match_map_1 = self.match_correlation(satellite, drone1)
        match_map_2 = self.match_correlation(satellite, drone2)
        match_map_3 = self.match_correlation(satellite, drone3)
        size = match_map_3.shape[-2:]

        match_map_1 = F.interpolate(
            match_map_1, size, mode="bilinear", align_corners=False
        )
        match_map_2 = F.interpolate(
            match_map_2, size, mode="bilinear", align_corners=False
        )
        match_map_3 = F.interpolate(
            match_map_3, size, mode="bilinear", align_corners=False
        )

        print("match_map_1", match_map_1.shape)
        print("match_map_2", match_map_2.shape)
        print("match_map_3", match_map_3.shape)

        normalized_weights = self.w_softmax(self.weights)

        fusion = (
            match_map_1 * normalized_weights[0]
            + match_map_2 * normalized_weights[1]
            + match_map_3 * normalized_weights[2]
        )

        # Sum along the channel dimension to get the final fused feature map
        fusion = fusion.sum(dim=1, keepdim=True)

        fusion = F.interpolate(
            fusion, size=self.upsample_size, mode="bilinear", align_corners=False
        )

        return fusion


class SaveLayerFeatures(nn.Module):
    """
    A helper module for saving the features of a layer during forward pass.

    """

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
    """
    A modified PVT (Pyramid Vision Transformer) model which saves features from
    its first three blocks during forward pass.

    Args:
        original_model: (nn.Module) The original PVT model to modify.

    """

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
    """
    Cross-View Localization model that uses a satellite and UAV (Unmanned Aerial Vehicle)
    view for localization.

    This model uses two modified PVT models for feature extraction from the satellite
    and UAV views, respectively. The extracted features are then passed through a Fusion
    module to produce a fused feature map.

    Args:
        satellite_resolution: (Tuple[int, int]) The resolution of the satellite images.

    """

    def __init__(self, satellite_resolution):
        super(CrossViewLocalizationModel, self).__init__()

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

    def forward(self, x_UAV, x_satellite):
        # Pytorch: [batch_size, channels, height, width]
        # numpy: [height, width, channels]

        feature_pyramid_UAV = self.feature_extractor_UAV(x_UAV)
        feature_pyramid_satellite = self.feature_extractor_satellite(x_satellite)
        last_sat_feature = feature_pyramid_satellite[0]

        fused_map = self.fusion(*feature_pyramid_UAV, last_sat_feature)

        fused_map = fused_map.squeeze(1)  # remove the unnecessary channel dimension

        return fused_map
