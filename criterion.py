#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


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
    def __init__(self, center_r=33, negative_weight=1, device="cuda"):
        super(HanningLoss, self).__init__()
        self.Center_R = center_r
        self.NG = negative_weight
        self.device = device

    def forward(self, preds, target):
        positive_samples = target > 0
        negative_samples = target == 0

        num_negative_samples = negative_samples.sum()
        num_positive_samples = positive_samples.sum()

        i_negative_weight = num_negative_samples / (
            num_positive_samples + num_negative_samples
        )

        negative_weights = self.NG / (i_negative_weight + 1)
        positive_weights = 1 / num_positive_samples

        preds = preds.squeeze(1)

        loss = (
            negative_weights
            * (preds[negative_samples] - target[negative_samples]).pow(2).sum()
        )

        loss2 = (
            positive_weights
            * (preds[positive_samples] - target[positive_samples]).pow(2).sum()
        )

        l = loss + loss2

        return l


class MSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, pred, true):
        return self.mse(torch.log1p(pred), torch.log1p(true))


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits, labels):
        logits = torch.sigmoid(logits)
        intersection = torch.sum(logits * labels)
        union = torch.sum(logits) + torch.sum(labels)
        dice_coeff = (2.0 * intersection + self.eps) / (union + self.eps)
        return 1 - dice_coeff


class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1e-3, alpha=0.01):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        C = self.theta * self.omega
        A = self.omega / (1 + torch.exp(-(y - self.epsilon) / self.alpha))
        diffs = A * (torch.log(1 + (delta_y / C).pow(self.alpha)))
        return diffs.mean()
