#!/usr/bin/env python3

# Pytorch: [batch_size, channels, height, width]
# numpy: [height, width, channels]

import torch.nn as nn
import timm
import torch
import torch.nn.functional as F


class Xcorr(nn.Module):
    """
    Cross-correlation module.
    """

    def __init__(self):
        super(Xcorr, self).__init__()

        self.batch_norm = nn.BatchNorm2d(1)

    def forward(self, query, search_map):
        """
        Compute the cross-correlation between a query and a search map.

        Parameters:
        query: A 4D tensor of shape (batch_size, channels, query_height, query_width).
        search_map: A 4D tensor of shape (batch_size, channels, search_map_height, search_map_width).

        Returns:
        corr_maps: A tensor of correlation maps.
        """

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

        search_map_padded = search_map_padded.reshape(1, bs * c, H, W)

        corr_maps = F.conv2d(search_map_padded, query, groups=bs)

        corr_maps = corr_maps.permute(1, 0, 2, 3)
        corr_maps = self.batch_norm(corr_maps)

        return corr_maps


class Fusion(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_size, fusion_dropout):
        """
        Fusion module which is a type of convolutional neural network.

        The Fusion class is designed to merge information from two separate
        input streams. In the case of a UAV (unmanned aerial vehicle) and
        SAT (satellite), the module uses a pyramid of features from both, and
        computes correlations between them to fuse them into a single output.
        This module utilizes several 1x1 convolutions and correlation layers
        for the fusion process. The fusion is controlled by learnable weights
        for each level of the pyramid.

        Args:
            in_channels (tuple): Tuple of 3 elements specifying the number of
            input channels for the 3 layers of the pyramid.
            out_channels (int): The number of output channels for the convolution operations.
            upsample_size (tuple): Tuple specifying the height and width for the output of the module.
        """

        super(Fusion, self).__init__()

        self.fusion_dropout = fusion_dropout

        if self.fusion_dropout is None:
            self.fusion_dropout = 0

        # UAV convolutions
        self.conv1_UAV = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels[0],
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                groups=out_channels,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=self.fusion_dropout),
        )
        self.conv2_UAV = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels[1],
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                groups=out_channels,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=self.fusion_dropout),
        )
        self.conv3_UAV = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels[2],
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                groups=out_channels,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=self.fusion_dropout),
        )

        # SAT convolutions
        self.conv1_SAT = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels[0],
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                groups=out_channels,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=self.fusion_dropout),
        )
        self.conv2_SAT = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels[1],
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                groups=out_channels,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=self.fusion_dropout),
        )
        self.conv3_SAT = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels[2],
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                groups=out_channels,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=self.fusion_dropout),
        )

        self.corrU1 = Xcorr()
        self.corrU2 = Xcorr()
        self.corrU3 = Xcorr()

        self.convU1_UAV = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                groups=64,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=self.fusion_dropout),
        )
        self.convU2_UAV = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                groups=64,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=self.fusion_dropout),
        )
        self.convU3_UAV = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                groups=64,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=self.fusion_dropout),
        )
        self.convU3_SAT = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                groups=64,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=self.fusion_dropout),
        )

        self.upsample_size = upsample_size
        self.fusion_weights = nn.Parameter(torch.ones(3))

    def forward(self, sat_feature_pyramid, UAV_feature_pyramid):
        """
        Perform the forward pass of the Fusion module.

        Args:
            sat_feature_pyramid (list of torch.Tensor): List of 3 tensors representing the satellite feature pyramid.
            Each tensor is of shape (batch_size, channels, height, width).
            UAV_feature_pyramid (list of torch.Tensor): List of 3 tensors representing the UAV feature pyramid.
            Each tensor is of shape (batch_size, channels, height, width).

        Returns:
            fused_map (torch.Tensor): The fused feature map resulting from the fusion of the input feature pyramids.
            The shape is (batch_size, channels, upsample_size[0], upsample_size[1]).
        """
        s1_UAV_feature = UAV_feature_pyramid[0]
        s2_UAV_feature = UAV_feature_pyramid[1]
        s3_UAV_feature = UAV_feature_pyramid[2]
        s1_sat_feature = sat_feature_pyramid[0]
        s2_sat_feature = sat_feature_pyramid[1]
        s3_sat_feature = sat_feature_pyramid[2]

        # UAV feature upsampling
        U1_UAV = self.conv1_UAV(s3_UAV_feature)
        U2_UAV = F.interpolate(
            U1_UAV, size=s2_UAV_feature.shape[-2:], mode="bicubic"
        ) + self.conv2_UAV(s2_UAV_feature)
        U3_UAV = F.interpolate(
            U2_UAV, size=s1_UAV_feature.shape[-2:], mode="bicubic"
        ) + self.conv3_UAV(s1_UAV_feature)

        # SAT feature upsampling
        U1_sat = self.conv1_SAT(s3_sat_feature)
        U2_sat = F.interpolate(
            U1_sat, size=s2_sat_feature.shape[-2:], mode="bicubic"
        ) + self.conv2_SAT(s2_sat_feature)
        U3_sat = F.interpolate(
            U2_sat, size=s1_sat_feature.shape[-2:], mode="bicubic"
        ) + self.conv3_SAT(s1_sat_feature)

        U1_UAV = self.convU1_UAV(U1_UAV)
        U2_UAV = self.convU2_UAV(U2_UAV)
        U3_UAV = self.convU3_UAV(U3_UAV)
        U3_sat = self.convU3_SAT(U3_sat)

        A1 = self.corrU1(U1_UAV, U3_sat)
        A2 = self.corrU2(U2_UAV, U3_sat)
        A3 = self.corrU3(U3_UAV, U3_sat)

        fw = self.fusion_weights / torch.sum(self.fusion_weights)

        fused_map = fw[0] * A1 + fw[1] * A2 + fw[2] * A3

        fused_map = F.interpolate(fused_map, size=self.upsample_size, mode="bicubic")

        return fused_map


class SaveLayerFeatures(nn.Module):
    def __init__(self):
        super(SaveLayerFeatures, self).__init__()
        self.outputs = None

    def forward(self, x):
        self.outputs = x.clone()
        return x

    def clear(self):
        self.outputs = None


class ModifiedPCPVT(nn.Module):
    """
    A modified PVT (Pyramid Vision Transformer) model which saves features from
    its first three blocks during forward pass.

    Args:
        original_model: (nn.Module) The original PVT model to modify.

    """

    def __init__(self, original_model, drops):
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
        self.save_l0 = SaveLayerFeatures()
        self.save_l1 = SaveLayerFeatures()
        self.save_l2 = SaveLayerFeatures()
        self.model.pos_block[0].proj.add_module("save_l0", self.save_l0)
        self.model.pos_block[1].proj.add_module("save_l1", self.save_l1)
        self.model.pos_block[2].proj.add_module("save_l2", self.save_l2)

        if drops is not None:
            self._set_dropout_values(self.model, drops)

    def _set_dropout_values(self, model, dropout_values):
        """
        Regulates the dropout values of the model.
        """
        for module in model.modules():
            if hasattr(module, "attn_drop"):
                module.attn_drop.p = dropout_values.get("attn_drop", module.attn_drop.p)
            if hasattr(module, "proj_drop"):
                module.proj_drop.p = dropout_values.get("proj_drop", module.proj_drop.p)
            if hasattr(module, "head_drop"):
                module.head_drop.p = dropout_values.get("head_drop", module.head_drop.p)
            if hasattr(module, "drop1"):
                module.drop1.p = dropout_values.get("mlp_drop1", module.drop1.p)
            if hasattr(module, "drop2"):
                module.drop2.p = dropout_values.get("mlp_drop2", module.drop2.p)
            if hasattr(module, "pos_drops"):
                for drop in module.pos_drops:
                    drop.p = dropout_values.get("pos_drops", drop.p)

    def forward(self, x):
        self.save_l0.clear()
        self.save_l1.clear()
        self.save_l2.clear()

        _ = self.model(x)

        return [  # Return the feature pyramids
            self.save_l0.outputs,
            self.save_l1.outputs,
            self.save_l2.outputs,
        ]


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

    def __init__(
        self,
        satellite_resolution,
        drops_UAV,
        drops_satellite,
        fusion_dropout,
        pretrained_twins,
    ):
        super(CrossViewLocalizationModel, self).__init__()

        self.satellite_resolution = satellite_resolution
        self.fusion_dropout = fusion_dropout

        if pretrained_twins is None:
            pretrained_twins = True

        # Feature extraction module
        self.backbone_UAV = timm.create_model(
            "twins_pcpvt_small", pretrained=pretrained_twins
        )
        self.feature_extractor_UAV = ModifiedPCPVT(self.backbone_UAV, drops_UAV)

        self.backbone_satellite = timm.create_model(
            "twins_pcpvt_small", pretrained=pretrained_twins
        )

        self.feature_extractor_satellite = ModifiedPCPVT(
            self.backbone_satellite, drops_satellite
        )

        self.fusion = Fusion(
            in_channels=[320, 128, 64],
            out_channels=64,
            upsample_size=self.satellite_resolution,
            fusion_dropout=self.fusion_dropout,
        )

    def forward(self, x_UAV, x_satellite):

        feature_pyramid_UAV = self.feature_extractor_UAV(x_UAV)
        feature_pyramid_satellite = self.feature_extractor_satellite(x_satellite)

        fus = self.fusion(feature_pyramid_satellite, feature_pyramid_UAV)

        fused_map = fus.squeeze(1)  # remove the unnecessary channel dimension

        return fused_map
