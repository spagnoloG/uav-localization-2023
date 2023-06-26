#!/usr/bin/env python3
import torch
import torch.nn as nn


class BalanceLoss(nn.Module):
    def __init__(self, w_neg=1.0, R=1, sat_image_size=512):
        super(BalanceLoss, self).__init__()
        self.w_neg = w_neg
        self.R = R
        self.sat_image_size = sat_image_size

    def forward(self, heatmap, label):
        """
        Compute the balance loss.
        """
        # heatmap = self.upscale_generated_heatmap(heatmap)

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

    def upscale_generated_heatmap(self, heatmap):
        """
        Upscale the generated heatmap to the size of the satellite image.
        """
        return nn.functional.interpolate(
            heatmap.unsqueeze(0),
            size=(self.sat_image_size, self.sat_image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
