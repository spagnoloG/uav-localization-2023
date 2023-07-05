#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


class BalanceLoss(nn.Module):
    """
    Balance loss implementation.

    Args:
        w_neg (float): Weight for negative samples.
        R (int): Threshold value.
        sat_image_size (int): Size of the satellite image.

    """

    def __init__(self, w_neg=1.0, R=1, sat_image_size=512):
        super(BalanceLoss, self).__init__()
        self.w_neg = w_neg
        self.R = R
        self.sat_image_size = sat_image_size

    def forward(self, heatmap, label):
        """
        Compute the balance loss.

        Args:
            heatmap (torch.Tensor): Predicted heatmap.
            label (torch.Tensor): Ground truth label.

        Returns:
            torch.Tensor: Computed balance loss.

        """
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
    """
    Hanning loss implementation.

    Args:
        center_r (int): Center radius.
        negative_weight (int): Weight for negative samples.
        device (str): Device to use for computation.

    """

    def __init__(self, center_r=33, negative_weight=1, device="cuda"):
        super(HanningLoss, self).__init__()
        self.Center_R = center_r
        self.NG = negative_weight
        self.device = device

    def forward(self, preds, target):
        """
        Compute the Hanning loss.

        Args:
            preds (torch.Tensor): Predicted tensor.
            target (torch.Tensor): Ground truth tensor.

        Returns:
            torch.Tensor: Computed Hanning loss.

        """
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
    """
    Mean Squared Logarithmic Error (MSLE) loss implementation.

    """

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
        """
        Compute the MSLE loss.

        Args:
            pred (torch.Tensor): Predicted tensor.
            true (torch.Tensor): Ground truth tensor.

        Returns:
            torch.Tensor: Computed MSLE loss.

        """
        logits = torch.sigmoid(logits)
        intersection = torch.sum(logits * labels)
        union = torch.sum(logits) + torch.sum(labels)
        dice_coeff = (2.0 * intersection + self.eps) / (union + self.eps)
        return 1 - dice_coeff


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing loss implementation.

    Args:
        omega (float): Omega value.
        theta (float): Theta value.
        epsilon (float): Epsilon value.
        alpha (float): Alpha value.

    """

    def __init__(self, omega=14, theta=0.5, epsilon=1e-3, alpha=0.01):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        """
        Compute the Adaptive Wing loss.

        Args:
            pred (torch.Tensor): Predicted tensor.
            target (torch.Tensor): Ground truth tensor.

        Returns:
            torch.Tensor: Computed Adaptive Wing loss.

        """
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        C = self.theta * self.omega
        A = self.omega / (1 + torch.exp(-(y - self.epsilon) / self.alpha))
        diffs = A * (torch.log(1 + (delta_y / C).pow(self.alpha)))
        return diffs.mean()


class WeightedLoss(nn.Module):
    """
    Weighted loss implementation.

    Args:
        image_size (int): Size of the image.
        negative_weight (float): Weight for negative samples.

    """

    def __init__(self, image_size=512, negative_weight=1.0):
        super(WeightedLoss, self).__init__()
        self.image_size = image_size
        self.negative_weight = negative_weight

    def forward(self, pred, target):
        """
        Compute the weighted loss.

        Args:
            pred (torch.Tensor): Predicted tensor.
            target (torch.Tensor): Ground truth tensor.

        Returns:
            torch.Tensor: Computed weighted loss.

        """
        # Count positive and negative labels
        positive_labels = (target > 0).float()
        negative_labels = (target <= 0).float()

        num_positive = positive_labels.sum()
        num_negative = negative_labels.sum()

        # Create Hanning Window for positive weights
        hanning_window = torch.hann_window(
            self.image_size, periodic=False, dtype=torch.float, device=pred.device
        )
        hanning_window = hanning_window.view(1, 1, -1, 1) * hanning_window.view(
            1, 1, 1, -1
        )
        positive_weights = positive_labels * hanning_window
        num_positive_weighted = positive_weights.sum()

        # Compute negative weights
        negative_weights = negative_labels * self.negative_weight / num_negative
        num_negative_weighted = negative_weights.sum()

        normalization = num_positive_weighted + num_negative_weighted

        # Assign weights
        weights = (positive_weights / normalization) + negative_weights / normalization

        # Compute weighted loss
        loss = F.binary_cross_entropy_with_logits(
            pred.squeeze(), target.squeeze(), weight=weights.squeeze()
        )
        return loss


class WeightedMSELoss(nn.Module):
    """
    Weighted Mean Squared Error (MSE) loss implementation.

    """

    def __init__(self):
        super().__init__()

    def forward(self, prediction, ground_truth):
        """
        Compute the weighted MSE loss.

        Args:
            prediction (torch.Tensor): Predicted tensor.
            ground_truth (torch.Tensor): Ground truth tensor.

        Returns:
            torch.Tensor: Computed weighted MSE loss.

        """
        mask = ground_truth == 0
        mse_loss = F.mse_loss(prediction, ground_truth, reduction="none")
        mse_loss[mask] = mse_loss[mask] / mse_loss.numel()
        return mse_loss.mean()
