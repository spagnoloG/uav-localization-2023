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
import numpy as np
from criterion import WeightedMSELoss


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

        self.criterion = WeightedMSELoss()

        self.prepare_dataloaders(config)

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
                        download_dataset=self.download_dataset,
                        heatmap_kernel_size=heatmap_kernel_size,
                        drone_view_patch_size=drone_view_patch_size,
                        metadata_rtree_index=self.metadata_rtree_index,
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
                subset_dataset = JoinedDataset(
                    dataset="test",
                    config=config,
                    download_dataset=self.download_dataset,
                    heatmap_kernel_size=heatmap_kernel_size,
                    drone_view_patch_size=drone_view_patch_size,
                    metadata_rtree_index=self.metadata_rtree_index,
                )
                self.val_dataloaders.append(
                    DataLoader(
                        subset_dataset,
                        batch_size=batch_size,
                        num_workers=self.num_workers,
                        shuffle=self.shuffle_dataset,
                    )
                )

            if self.metadata_rtree_index is None:
                self.metadata_rtree_index = (
                    subset_dataset.metadata_rtree_index
                    if self.val_subset_size is None
                    else subset_dataset.dataset.metadata_rtree_index
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
                    sat_infos,
                    heatmap_gt,
                ) in tqdm(
                    enumerate(dataloader),
                    total=len(dataloader),
                ):
                    drone_images = drone_images.to(self.device)
                    sat_images = sat_images.to(self.device)
                    heatmap_gt = heatmap_gt.to(self.device)
                    # Forward pass
                    outputs = self.model(drone_images, sat_images)
                    # Calculate loss
                    loss = self.criterion(outputs, heatmap_gt)
                    # Accumulate the loss
                    running_loss += loss.item() * drone_images.size(0)

                    if self.plot:
                        self.plot_results(
                            drone_images[0], sat_images[0], heatmap_gt[0], outputs[0], i
                        )
                total_samples += len(dataloader)

        epoch_loss = running_loss / total_samples
        logger.info("Validation Loss: {:.4f}".format(epoch_loss))

    def visualize_model(self):
        tensor_uav = torch.randn(1, 128, 128, 3)
        tensor_sat = torch.randn(1, 512, 512, 3)

        fused_heatmap = self.model(tensor_uav, tensor_sat)

        dot = make_dot(fused_heatmap, params=dict(self.model.named_parameters()))
        dot.format = "png"
        os.makedirs("./vis", exist_ok=True)
        dot.render("model", "./vis", view=True)

    def plot_results(self, drone_image, sat_image, heatmap_gt, heatmap_pred, i):
        """
        Plot the validation results.

        This function will plot the validation results for the specified number
        of epochs.
        """

        # Plot them on the same figure
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(1, 4, 1)
        img = drone_image.permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)  # Ensure all values are within [0, 1]
        ax.imshow(img)
        ax.set_title("Drone Image")
        ax.axis("off")

        ax = fig.add_subplot(1, 4, 2)
        img = sat_image.permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title("Satellite Image")
        ax.axis("off")

        ax = fig.add_subplot(1, 4, 3)
        heatmap = heatmap_gt.squeeze(0).cpu().numpy()
        ax.imshow(heatmap, cmap="viridis")
        ax.set_title("Ground Truth Heatmap")
        ax.axis("off")

        ax = fig.add_subplot(1, 4, 4)
        heatmap = heatmap_pred.squeeze(0).cpu().numpy()
        ax.imshow(heatmap, cmap="viridis")
        ax.set_title("Predicted Heatmap")
        ax.axis("off")

        os.makedirs(f"./vis/{self.val_hash}", exist_ok=True)

        plt.savefig(f"./vis/{self.val_hash}/validation_{self.val_hash}-{i}.png")

        plt.close()
        plt.clf()


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
    # validator.visualize_model()


if __name__ == "__main__":
    main()
