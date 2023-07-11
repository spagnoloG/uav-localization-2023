#!/usr/bin/env python3

# Pytorch: [batch_size, channels, height, width]
# numpy: [height, width, channels]
# [batch_size, height, width, channels] -> [batch_size, channels, height, width]

import torch.nn as nn
import timm
import torch
import torch.nn.functional as F


class Fusion(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_size):
        super(Fusion, self).__init__()
        self.conv1_UAV = nn.Conv2d(
            in_channels=in_channels[0], out_channels=out_channels, kernel_size=1
        )
        self.conv2_UAV = nn.Conv2d(
            in_channels=in_channels[1], out_channels=out_channels, kernel_size=1
        )
        self.conv3_UAV = nn.Conv2d(
            in_channels=in_channels[2], out_channels=out_channels, kernel_size=1
        )

        self.conv1_SAT = nn.Conv2d(
            in_channels=in_channels[0], out_channels=out_channels, kernel_size=1
        )
        self.conv2_SAT = nn.Conv2d(
            in_channels=in_channels[1], out_channels=out_channels, kernel_size=1
        )
        self.conv3_SAT = nn.Conv2d(
            in_channels=in_channels[2], out_channels=out_channels, kernel_size=1
        )

        self.corrU1 = Correlation()
        self.corrU2 = Correlation()
        self.corrU3 = Correlation()

        self.upsample_size = upsample_size
        self.fusion_weights = nn.Parameter(torch.ones(3))

    def forward(self, sat_feature_pyramid, UAV_feature_pyramid):
        s1_drone_feature = UAV_feature_pyramid[0]
        s2_drone_feature = UAV_feature_pyramid[1]
        s3_drone_feature = UAV_feature_pyramid[2]
        s1_sat_feature = sat_feature_pyramid[0]
        s2_sat_feature = sat_feature_pyramid[1]
        s3_sat_feature = sat_feature_pyramid[2]
        U1_drone = self.conv1_UAV(s3_drone_feature)
        U2_drone = F.interpolate(
            U1_drone, size=s2_drone_feature.shape[-2:]
        ) + self.conv2_UAV(s2_drone_feature)
        U3_drone = F.interpolate(
            U2_drone, size=s1_drone_feature.shape[-2:]
        ) + self.conv3_UAV(s1_drone_feature)
        U1_sat = self.conv1_SAT(s3_sat_feature)
        U2_sat = F.interpolate(U1_sat, size=s2_sat_feature.shape[-2:]) + self.conv2_SAT(
            s2_sat_feature
        )
        U3_sat = F.interpolate(U2_sat, size=s1_sat_feature.shape[-2:]) + self.conv3_SAT(
            s1_sat_feature
        )
        A1 = self.corrU1(U1_drone, U3_sat)
        A2 = self.corrU2(U2_drone, U3_sat)
        A3 = self.corrU3(U3_drone, U3_sat)

        fw = F.softmax(self.fusion_weights, dim=0)

        fused_map = fw[0] * A1 + fw[1] * A2 + fw[2] * A3

        fused_map = F.interpolate(
            fused_map, size=self.upsample_size, mode="bilinear", align_corners=True
        )

        return fused_map


class Correlation(nn.Module):
    """Module to compute correlation between a query tensor and a search map tensor."""

    def __init__(self):
        super(Correlation, self).__init__()
        self.batch_norm = nn.BatchNorm2d(1)

    def forward(self, query, search_map):
        """
        Compute the correlation between a query and a search map.

        Parameters:
        query: A 4D tensor of shape (batch_size, channels, query_height, query_width).
        search_map: A 4D tensor of shape (batch_size, channels, search_map_height, search_map_width).

        Returns:
        corr_maps: A tensor of correlation maps.
        """
        # Check if the inputs are 4D tensors.
        if not (query.dim() == search_map.dim() == 4):
            raise ValueError("Both query and search_map need to be 4D tensors")

        # Check if search_map has larger or equal spatial dimensions than query.
        if not (
            search_map.shape[2] >= query.shape[2]
            and search_map.shape[3] >= query.shape[3]
        ):
            raise ValueError(
                "Each spatial dimension of search_map must be larger or equal to that of query"
            )

        # Group convolution as correlation
        # Pad search map to maintain spatial resolution
        search_map_padded = F.pad(
            search_map,
            (
                query.shape[3] // 2,
                query.shape[3] // 2,
                query.shape[2] // 2,
                query.shape[2] // 2,
            ),
        )

        bs, c, h, w = query.shape
        _, _, H, W = search_map_padded.shape

        corr_maps = []
        for map_, q_ in zip(search_map_padded.split(1), query.split(1)):
            corr_map = F.conv2d(map_, q_, groups=1)
            corr_maps.append(corr_map)

        # Concatenate the correlation maps along the batch dimension.
        corr_maps = torch.cat(corr_maps, dim=0)

        corr_maps = self.batch_norm(corr_maps)

        return corr_maps


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

        self.fusion = Fusion(
            in_channels=[320, 128, 64],
            out_channels=64,
            upsample_size=self.satellite_resolution,
        )

    def forward(self, x_UAV, x_satellite):
        # Pytorch: [batch_size, channels, height, width]
        # numpy: [height, width, channels]

        feature_pyramid_UAV = self.feature_extractor_UAV(x_UAV)
        feature_pyramid_satellite = self.feature_extractor_satellite(x_satellite)

        fus = self.fusion(feature_pyramid_satellite, feature_pyramid_UAV)

        fused_map = fus.squeeze(1)  # remove the unnecessary channel dimension

        return fused_map
