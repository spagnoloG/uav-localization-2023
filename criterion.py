#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class HanningLoss(nn.Module):
    def __init__(self, center_r=33, negative_weight=1.0, device="cuda"):
        super(HanningLoss, self).__init__()
        self.Center_R = center_r
        self.NG = negative_weight
        self.device = device

        self.positive_weight = self.compute_positive_weight()

    def compute_positive_weight(self):
        hann1d = torch.hann_window(self.Center_R).to(self.device)
        hanning_window = hann1d.unsqueeze(1) * hann1d.unsqueeze(0)
        return hanning_window.mean()  # single weight for all positive samples

    def forward(self, preds, target):
        positive_samples = target == 1
        negative_samples = target == 0

        NN = negative_samples.sum()
        NW = positive_samples.sum()

        negative_weights = self.NG / (NN * (NW + 1))

        preds = preds.squeeze(1)

        loss = (
            negative_weights
            * (preds[negative_samples] - target[negative_samples]).pow(2).sum()
        )

        loss += (
            self.positive_weight
            * (preds[positive_samples] - target[positive_samples]).pow(2)
        ).sum()

        return loss
