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
from criterion import HanningLoss, RDS
from map_utils import MapUtils
import numpy as np
import matplotlib.patches as patches
import rasterio
import json
import logging

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
        self.shuffle_dataset = self.config["val"]["shuffle_dataset"]
        self.val_dataloaders = []
        self.batch_size = config["val"]["batch_size"]
        self.heatmap_kernel_size = config["dataset"]["heatmap_kernel_size"]
        self.loss_fn = config["train"]["loss_fn"]
        self.RDS = RDS()

        if self.loss_fn == "hanning":
            self.criterion = HanningLoss(
                kernel_size=self.heatmap_kernel_size, device=self.device
            )
        elif self.loss_fn == "mse":
            self.criterion = torch.nn.MSELoss(reduction="mean")

        elif self.loss_fn == "cwmse":
            self.criterion = CrossWeightedMSE()
            self.config["dataset"]["heatmap_type"] = "gaussian"
        else:
            raise NotImplementedError(
                f"Loss function {self.loss_fn} is not implemented"
            )

        self.prepare_dataloaders(config)
        self.map_utils = MapUtils()

        self.load_model()
        self.update_log_filepath(
                f"./checkpoints/{self.val_hash}/validation.log"
        )

    def update_log_filepath(self, log_filepath):
        """
        Update the log file path.

        log_filepath: the new log file path
        """
        global logger

        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)

        updated_handler = logging.FileHandler(log_filepath)
        logger.addHandler(updated_handler)
        format_str = "[%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s] -> %(message)s"
        formatter = logging.Formatter(format_str)
        updated_handler.setFormatter(formatter)

        logger.addHandler(updated_handler)

    def prepare_dataloaders(self, config):
        if self.val_subset_size is not None:
            logger.info(f"Using val subset of size {self.val_subset_size}")
            subset_dataset = torch.utils.data.Subset(
                JoinedDataset(
                    dataset="test",
                    config=config,
                ),
                indices=range(self.val_subset_size),
            )
            self.val_dataloader = DataLoader(
                subset_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=self.shuffle_dataset,
            )
        else:
            logger.info("Using full val dataset")
            subset_dataset = JoinedDataset(
                dataset="test",
                config=config,
            )
            self.val_dataloader = DataLoader(
                subset_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=self.shuffle_dataset,
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
                    self.config["dataset"]["sat_patch_w"],
                    self.config["dataset"]["sat_patch_h"],
                )
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
        running_RDS = 0.0
        with torch.no_grad():
            for i, (
                drone_images,
                drone_infos,
                sat_images,
                heatmaps_gt,
            ) in tqdm(
                enumerate(self.val_dataloader),
                total=len(self.val_dataloader),
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

                x_sat = drone_infos["x_sat"]
                y_sat = drone_infos["y_sat"]

                ### RDS ###
                running_RDS += self.RDS(
                    outputs,
                    x_sat,
                    y_sat,
                    heatmaps_gt[0].shape[-1],
                    heatmaps_gt[0].shape[-2],
                )
                ### RDS ###
                if self.plot:
                    for j in range(len(outputs)):
                        metadata = {
                            "x_sat": drone_infos["x_sat"][j].item(),
                            "y_sat": drone_infos["y_sat"][j].item(),
                            "x_offset": drone_infos["x_offset"][j].item(),
                            "y_offset": drone_infos["y_offset"][j].item(),
                            "zoom_level": drone_infos["zoom_level"][j].item(),
                            "lat_gt": drone_infos["coordinate"]["latitude"][j].item(),
                            "lon_gt": drone_infos["coordinate"]["longitude"][j].item(),
                            "filename": drone_infos["filename"][j],
                            "scale": drone_infos["scale"][j].item(),
                        }

                        self.plot_results(
                            drone_images[j].detach(),
                            sat_images[j].detach(),
                            heatmap_gt[j].detach(),
                            outputs[j].detach(),
                            metadata,
                            i,
                            j,
                        )

            total_samples += len(self.val_dataloader)

        epoch_loss = running_loss / total_samples

        self.val_loss = epoch_loss

        logger.info(f"Validation loss: {epoch_loss}")
        logger.info(f"Validation RDS: {running_RDS.cpu().item() / total_samples}")

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
        metadata,
        i,
        j,
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
                            self.config["dataset"]["mean"],
                            self.config["dataset"]["std"],
                        )
                    ],
                    std=[1 / s for s in self.config["dataset"]["std"]],
                ),
                transforms.ToPILImage(),
            ]
        )

        # Compute prediction, ground truth positions, and the distance
        heatmap_pred_np = heatmap_pred.cpu().numpy()
        y_pred, x_pred = np.unravel_index(
            np.argmax(heatmap_pred_np), heatmap_pred_np.shape
        )

        sat_image_path = metadata["filename"]
        zoom_level = metadata["zoom_level"]
        x_offset = metadata["x_offset"]
        y_offset = metadata["y_offset"]

        with rasterio.open(f"{sat_image_path}_sat_{zoom_level}.tiff") as s_image:
            sat_transform = s_image.transform
            lon_pred, lat_pred = rasterio.transform.xy(
                sat_transform, y_pred + y_offset, x_pred + x_offset
            )

        metadata["lat_pred"] = lat_pred
        metadata["lon_pred"] = lon_pred

        metadata["rds"] = self.map_utils.RDS(
            10,
            np.abs(metadata["x_sat"] - x_pred),
            np.abs(metadata["y_sat"] - y_pred),
            heatmap_gt.shape[-1],
            heatmap_gt.shape[-2],
        )

        # Initialize figure
        fig, axs = plt.subplots(3, 2, figsize=(20, 30))

        # Subplot 1: Drone Image
        axs[0, 0].imshow(inverse_transforms(drone_image))
        axs[0, 0].set_title("Drone Image")
        axs[0, 0].axis("off")

        # Subplot 2: Satellite Image
        axs[0, 1].imshow(inverse_transforms(sat_image))
        axs[0, 1].set_title("Satellite Image")
        axs[0, 1].axis("off")

        # Subplot 3: Ground Truth Heatmap
        im3 = axs[1, 0].imshow(heatmap_gt.squeeze(0).cpu().numpy(), cmap="viridis")
        axs[1, 0].set_title(
            f"Ground Truth Heatmap, Latitute: {metadata['lat_gt']}, Longitude: {metadata['lon_gt']}"
        )
        axs[1, 0].axis("off")
        fig.colorbar(im3, ax=axs[1, 0])

        # Subplot 4: Predicted Heatmap
        im4 = axs[1, 1].imshow(heatmap_pred.squeeze(0).cpu().numpy(), cmap="viridis")
        axs[1, 1].set_title(
            "Predicted Heatmap, Latitute: {metadata['lat_pred']}, Longitude: {metadata['lon_pred']}"
        )
        axs[1, 1].axis("off")
        fig.colorbar(im4, ax=axs[1, 1])

        # Subplot 5: Satellite Image with Predicted Heatmap and circles
        axs[2, 0].imshow(inverse_transforms(sat_image))
        im5 = axs[2, 0].imshow(
            heatmap_pred.squeeze(0).cpu().numpy(), cmap="jet", alpha=0.55
        )
        pred_circle = patches.Circle(
            (x_pred, y_pred), radius=10, edgecolor="blue", facecolor="none", linewidth=4
        )
        gt_circle = patches.Circle(
            (metadata["x_sat"], metadata["y_sat"]),
            radius=10,
            edgecolor="red",
            facecolor="none",
            linewidth=4,
        )
        axs[2, 0].add_patch(pred_circle)
        axs[2, 0].add_patch(gt_circle)
        axs[2, 0].set_title("Satellite Image with Predicted Heatmap")
        axs[2, 0].legend(
            [pred_circle, gt_circle], ["Prediction", "Ground Truth"], loc="upper right"
        )
        axs[2, 0].axis("off")

        # Subplot 6: Satellite Image with Ground Truth Heatmap
        axs[2, 1].imshow(inverse_transforms(sat_image))
        im6 = axs[2, 1].imshow(
            heatmap_gt.squeeze(0).cpu().numpy(), cmap="jet", alpha=0.55
        )
        axs[2, 1].set_title("Satellite Image with Ground Truth Heatmap")
        axs[2, 1].axis("off")

        # Add metadata as text
        metadata_text = f'Filename: {metadata["filename"]}\nZoom Level: {metadata["zoom_level"]}\nRDS: {metadata["rds"]}\nDrone image scale: {metadata["scale"]}'
        fig.text(0.5, 0.05, metadata_text, ha="center", fontsize=16)

        # Save the figure
        os.makedirs(f"./vis/{self.val_hash}", exist_ok=True)
        plt.savefig(f"./vis/{self.val_hash}/validation_{self.val_hash}-{i}-{j}.png")
        plt.close()

        with open(
            f"./vis/{self.val_hash}/validation_{self.val_hash}-{i}-{j}.json", "w"
        ) as f:
            json.dump(metadata, f)


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
