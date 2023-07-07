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
        self.match_batchnorm = nn.BatchNorm2d(1)

        # self.conv1 = nn.Conv2d(
        #    in_channels=in_channels[0], out_channels=out_channels, kernel_size=1
        # )
        # self.conv2 = nn.Conv2d(
        #    in_channels=in_channels[1], out_channels=out_channels, kernel_size=1
        # )
        # self.conv3 = nn.Conv2d(
        #    in_channels=in_channels[2], out_channels=out_channels, kernel_size=1
        # )

        # Dodaj padding pri conv correlaciji
        # self.weights = nn.Parameter(torch.ones(3, 1, 1, 1))
        # self.w_softmax = nn.Softmax(dim=0)

        # self.match_correlation = MatchCorrelation()

    def match_corr(self, drone_feature, satellite_feature):
        b, c, h, w = satellite_feature.shape
        # Here the correlation layer is implemented using a trick with the
        # conv2d function using groups in order to do the correlation with
        # batch dimension. Basically we concatenate each element of the batch
        # in the channel dimension for the search image (making it
        # [1 x (B.C) x H' x W']) and setting the number of groups to the size of
        # the batch. This grouped convolution/correlation is equivalent to a
        # correlation between the two images, though it is not obvious.
        match_map = F.conv2d(
            satellite_feature.view(1, b * c, h, w), drone_feature, groups=b
        )
        # Here we reorder the dimensions to get back the batch dimension.
        match_map = match_map.permute(1, 0, 2, 3)
        match_map = self.match_batchnorm(match_map)
        return match_map

    def forward(self, satellite_feature, drone_feature):
        fusion = self.match_corr(drone_feature, satellite_feature)
        # print("match map shape: ", mm.shape)
        # print("sat feature shape: ", satellite_feature.shape)
        # print("drone feature shape", drone_feature.shape)

        # Up-sample the fusion map to the original size of the satellite image
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
        last_sat_feature = feature_pyramid_satellite[-1]
        last_drone_feature = feature_pyramid_UAV[-1]

        fused_map = self.fusion(last_sat_feature, last_drone_feature)

        fused_map = fused_map.squeeze(1)  # remove the unnecessary channel dimension

        return fused_map
