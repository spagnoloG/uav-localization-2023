#!/usr/bin/env python3
from logger import logger
from joined_dataset import JoinedDataset
import torch
from torch.utils.data import DataLoader
from model import CrossViewLocalizationModel
import os
from tqdm import tqdm
import yaml
import argparse
from torchviz import make_dot
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from criterion import JustAnotherWeightedMSELoss
from map_utils import MapUtils
import numpy as np
import matplotlib.patches as patches


class CrossViewValidator:
    """Validator class for cross-view (UAV and satellite) image learning"""

    def __init__(self, config=None):
        """Initialize the validator class

        device: the device to run the validation on
        batch_size: the batch size to use for validation
        num_workers: the number of workers to use for validation
        plot: whether to plot the validation results
        val_hash: the weights of the model to use for validation
        """

        self.config = config
        self.device = self.config["val"]["device"]
        self.num_workers = self.config["val"]["num_workers"]
        self.plot = self.config["val"]["plot"]
        self.val_hash = self.config["val"]["checkpoint_hash"]
        self.val_subset_size = self.config["val"]["val_subset_size"]
        self.download_dataset = self.config["val"]["download_dataset"]
        self.batch_sizes = self.config["val"]["batch_sizes"]
        self.heatmap_kernel_sizes = self.config["val"]["heatmap_kernel_sizes"]
        self.drone_view_patch_sizes = self.config["val"]["drone_view_patch_sizes"]
        self.shuffle_dataset = self.config["val"]["shuffle_dataset"]
        self.val_dataloaders = []

        self.criterion = JustAnotherWeightedMSELoss()

        self.prepare_dataloaders(config)
        self.map_utils = MapUtils()

        self.load_model()

    def prepare_dataloaders(self, config):
        self.metadata_rtree_index = None

        for batch_size, heatmap_kernel_size, drone_view_patch_size in zip(
            self.batch_sizes, self.heatmap_kernel_sizes, self.drone_view_patch_sizes
        ):
            if self.val_subset_size is not None:
                logger.info(f"Using val subset of size {self.val_subset_size}")
                subset_dataset = torch.utils.data.Subset(
                    JoinedDataset(
                        dataset="test",
                        config=config,
                        heatmap_kernel_size=heatmap_kernel_size,
                        drone_patch_h=drone_view_patch_size,
                        drone_patch_w=drone_view_patch_size,
                    ),
                    indices=range(self.val_subset_size),
                )
                self.val_dataloaders.append(
                    DataLoader(
                        subset_dataset,
                        batch_size=batch_size,
                        num_workers=self.num_workers,
                        shuffle=self.shuffle_dataset,
                    )
                )
            else:
                logger.info("Using full val dataset")
                subset_dataset = JoinedDataset(
                    dataset="test",
                    config=config,
                    heatmap_kernel_size=heatmap_kernel_size,
                    drone_patch_h=drone_view_patch_size,
                    drone_patch_w=drone_view_patch_size,
                )
                self.val_dataloaders.append(
                    DataLoader(
                        subset_dataset,
                        batch_size=batch_size,
                        num_workers=self.num_workers,
                        shuffle=self.shuffle_dataset,
                    )
                )

    def load_model(self):
        """
        Load the model for the validation phase.

        This function will load the model state dict from a saved checkpoint.
        The epoch used for loading the model is stored in the config file.
        """

        epoch = self.config["val"]["checkpoint_epoch"]

        # construct the path of the saved checkpoint
        load_path = f"./checkpoints/{self.val_hash}/checkpoint-{epoch}.pt"

        if not os.path.isfile(load_path):
            logger.error(f"No checkpoint found at '{load_path}'")
            raise FileNotFoundError(f"No checkpoint found at '{load_path}'")

        checkpoint = torch.load(load_path)

        self.model = torch.nn.DataParallel(
            CrossViewLocalizationModel(
                satellite_resolution=(
                    self.config["sat_dataset"]["patch_w"],
                    self.config["sat_dataset"]["patch_h"],
                ),
            )
        )
        # load the state dict into the model
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # move the model to the correct device
        self.model.to(self.device)

        logger.info(f"Model loaded from '{load_path}' for validation.")

    def run_validation(self):
        """
        Run the validation phase.

        This function will run the validation phase for the specified number of
        epochs. The validation loss will be printed after each epoch.
        """
        logger.info("Starting validation...")

        self.validate()

        logger.info("Validation done.")

    def validate(self):
        """
        Perform one epoch of validation.
        """
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for dataloader, h_kernel_size, d_patch_size in zip(
                self.val_dataloaders,
                self.heatmap_kernel_sizes,
                self.drone_view_patch_sizes,
            ):
                logger.info(
                    f"Validating epoch with kernel size {h_kernel_size} and patch size {d_patch_size}"
                )
                for i, (
                    drone_images,
                    drone_infos,
                    sat_images,
                    heatmaps_gt,
                    x_sat,
                    y_sat,
                ) in tqdm(
                    enumerate(dataloader),
                    total=len(dataloader),
                ):
                    drone_images = drone_images.to(self.device)
                    sat_images = sat_images.to(self.device)
                    heatmap_gt = heatmaps_gt.to(self.device)
                    # Forward pass
                    outputs = self.model(drone_images, sat_images)
                    # Calculate loss
                    loss = self.criterion(outputs, heatmap_gt)
                    # Accumulate the loss
                    running_loss += loss.item() * drone_images.size(0)

                    if self.plot:
                        lat_gt, lon_gt = (
                            drone_infos["coordinate"]["latitude"][0].item(),
                            drone_infos["coordinate"]["longitude"][0].item(),
                        )
                        self.plot_results(
                            drone_images[0].detach(),
                            sat_images[0].detach(),
                            heatmap_gt[0].detach(),
                            outputs[0].detach(),
                            lat_gt,
                            lon_gt,
                            x_sat[0].item(),
                            y_sat[0].item(),
                            i,
                        )

                total_samples += len(dataloader)

        epoch_loss = running_loss / total_samples

        self.val_loss = epoch_loss

        logger.info(f"Validation loss: {epoch_loss}")

    def visualize_model(self):
        tensor_uav = torch.randn(1, 128, 128, 3)
        tensor_sat = torch.randn(1, 512, 512, 3)

        fused_heatmap = self.model(tensor_uav, tensor_sat)

        dot = make_dot(fused_heatmap, params=dict(self.model.named_parameters()))
        dot.format = "png"
        os.makedirs("./vis", exist_ok=True)
        dot.render("model", "./vis", view=True)

    def plot_results(
        self,
        drone_image,
        sat_image,
        heatmap_gt,
        heatmap_pred,
        lat_gt,
        lon_gt,
        x_gt,
        y_gt,
        i,
    ):
        """
        Plot the validation results.

        This function will plot the validation results for the specified number of epochs.
        """
        # Inverse transform for images
        inverse_transforms = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[
                        -m / s
                        for m, s in zip(
                            self.config["sat_dataset"]["mean"],
                            self.config["sat_dataset"]["std"],
                        )
                    ],
                    std=[1 / s for s in self.config["sat_dataset"]["std"]],
                ),
                transforms.ToPILImage(),
            ]
        )

        # Compute prediction, ground truth positions, and the distance
        heatmap_pred_np = heatmap_pred.cpu().numpy()
        y_pred, x_pred = np.unravel_index(
            np.argmax(heatmap_pred_np), heatmap_pred_np.shape
        )

        # Initialize figure
        fig = plt.figure(figsize=(20, 20))

        # Subplot 1: Drone Image
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.imshow(inverse_transforms(drone_image))
        ax1.set_title("Drone Image")
        ax1.axis("off")

        # Subplot 2: Satellite Image
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.imshow(inverse_transforms(sat_image))
        ax2.set_title("Satellite Image")
        ax2.axis("off")

        # Subplot 3: Ground Truth Heatmap
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.imshow(heatmap_gt.squeeze(0).cpu().numpy(), cmap="viridis")
        ax3.set_title("Ground Truth Heatmap")
        ax3.axis("off")

        # Subplot 4: Predicted Heatmap
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.imshow(heatmap_pred.squeeze(0).cpu().numpy(), cmap="viridis")
        ax4.set_title("Predicted Heatmap")
        ax4.axis("off")

        # Subplot 5: Satellite Image with Predicted Heatmap and circles
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.imshow(inverse_transforms(sat_image))
        ax5.imshow(heatmap_pred.squeeze(0).cpu().numpy(), cmap="jet", alpha=0.55)
        ax5.add_patch(
            patches.Circle(
                (x_pred, y_pred),
                radius=10,
                edgecolor="blue",
                facecolor="none",
                linewidth=4,
            )
        )
        ax5.add_patch(
            patches.Circle(
                (x_gt, y_gt), radius=10, edgecolor="red", facecolor="none", linewidth=4
            )
        )
        ax5.set_title("Satellite Image with Predicted Heatmap")
        ax5.legend(["Prediction", "Ground Truth"], loc="upper right")
        ax5.axis("off")

        # Subplot 6: Satellite Image with Ground Truth Heatmap
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.imshow(inverse_transforms(sat_image))
        ax6.imshow(heatmap_gt.squeeze(0).cpu().numpy(), cmap="jet", alpha=0.55)
        ax6.set_title("Satellite Image with Ground Truth Heatmap")
        ax6.axis("off")  # Save the figure
        os.makedirs(f"./vis/{self.val_hash}", exist_ok=True)
        plt.savefig(f"./vis/{self.val_hash}/validation_{self.val_hash}-{i}.png")
        plt.close()


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Run the validation phase for the cross-view model."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configuration",
        help="The path to the configuration file.",
    )

    args = parser.parse_args()

    config = load_config(f"./conf/{args.config}.yaml")

    validator = CrossViewValidator(config=config)

    validator.run_validation()


if __name__ == "__main__":
    main()
