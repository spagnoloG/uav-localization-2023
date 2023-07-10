#!/usr/bin/env python3
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from joined_dataset import JoinedDataset
from torch.utils.data import DataLoader
from logger import logger
import hashlib
import datetime
import os
import matplotlib.pyplot as plt
from model import CrossViewLocalizationModel
import yaml
import argparse
from criterion import WeightedMSELoss


class CrossViewTrainer:
    """Trainer class for cross-view (UAV and satellite) image learning"""

    def __init__(
        self,
        device,
        criterion,
        lr=3e-4,
        weight_decay=5e-4,
        batch_sizes=[2],
        num_workers=4,
        num_epochs=16,
        shuffle_dataset=True,
        checkpoint_hash=None,
        checkpoint_epoch=None,
        train_subset_size=None,
        val_subset_size=None,
        plot=False,
        config=None,
    ):
        """
        Initialize the CrossViewTrainer.

        backbone: the pretrained DeiT-S model, with its classifier removed
        device: the device to train on
        criterion: the loss function to use
        lr: learning rate
        weight_decay: weight decay
        batch_size: batch size
        num_workers: number of threads to use for the dataloader
        num_epochs: number of epochs to train for
        shuffle_dataset: whether to shuffle the dataset
        checkpoint_hash: the hash of the checkpoint to load
        checkpoint_epoch: the epoch of the checkpoint to load
        train_subset_size: the size of the train subset to use
        val_subset_size: the size of the val subset to use
        plot: whether to plot the intermediate results of the model
        config: the confguration file
        """
        self.device = config["train"]["device"]
        self.criterion = criterion
        self.lr_backbone = config["train"]["lr_backbone"]
        self.lr_fusion = self.lr_backbone * 1.5
        self.weight_decay = config["train"]["weight_decay"]
        self.num_workers = config["train"]["num_workers"]
        self.num_epochs = config["train"]["num_epochs"]
        self.shuffle_dataset = config["train"]["shuffle_dataset"]
        self.checkpoint_hash = config["train"]["checkpoint_hash"]
        self.checkpoint_epoch = config["train"]["checkpoint_epoch"]
        self.train_subset_size = config["train"]["train_subset_size"]
        self.val_subset_size = config["train"]["val_subset_size"]
        self.plot = config["train"]["plot"]
        self.current_epoch = 0
        self.download_dataset = config["train"]["download_dataset"]
        self.milestones = config["train"]["milestones"]
        self.batch_sizes = config["train"]["batch_sizes"]
        self.heatmap_kernel_sizes = config["train"]["heatmap_kernel_sizes"]
        self.drone_view_patch_sizes = config["train"]["drone_view_patch_sizes"]
        self.train_dataloaders = []
        self.val_dataloaders = []

        self.prepare_dataloaders(config)

        self.model = torch.nn.DataParallel(
            CrossViewLocalizationModel(
                satellite_resolution=(
                    config["sat_dataset"]["patch_w"],
                    config["sat_dataset"]["patch_h"],
                ),
            )
        )

        self.params_to_update_backbone = list(
            self.model.module.feature_extractor_UAV.parameters()
        ) + list(self.model.module.feature_extractor_satellite.parameters())
        self.params_to_update_fusion = list(self.model.module.fusion.parameters())

        self.optimizer = AdamW(
            [
                {"params": self.params_to_update_backbone, "lr": self.lr_backbone},
                {"params": self.params_to_update_fusion, "lr": self.lr_fusion},
            ],
            lr=self.lr_backbone,
        )

        self.scheduler = MultiStepLR(
            self.optimizer, milestones=self.milestones, gamma=0.1
        )

        self.model.to(self.device)

        if self.checkpoint_hash is not None and self.checkpoint_epoch is not None:
            try:
                self.current_epoch = self.load_checkpoint(
                    self.checkpoint_hash, self.checkpoint_epoch
                )
            except FileNotFoundError:
                logger.error(
                    f"Checkpoint with hash {self.checkpoint_hash} not found. Starting from scratch."
                )

        else:
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d-%H-%M-%S")
            now_hash = hashlib.sha1(now_str.encode()).hexdigest()
            self.checkpoint_hash = now_hash

        logger.info(
            f"Using chekpoint hash {self.checkpoint_hash}, starting from epoch {self.current_epoch}"
        )

    def prepare_dataloaders(self, config):
        self.metadata_rtree_index = None

        for batch_size, heatmap_kernel_size, drone_view_patch_size in zip(
            self.batch_sizes, self.heatmap_kernel_sizes, self.drone_view_patch_sizes
        ):
            if self.train_subset_size is not None:
                logger.info(f"Using train subset of size {self.train_subset_size}")
                subset_dataset = torch.utils.data.Subset(
                    JoinedDataset(
                        dataset="train",
                        config=config,
                        download_dataset=self.download_dataset,
                        heatmap_kernel_size=heatmap_kernel_size,
                        drone_view_patch_size=drone_view_patch_size,
                        metadata_rtree_index=self.metadata_rtree_index,
                    ),
                    indices=range(self.train_subset_size),
                )
                self.train_dataloaders.append(
                    DataLoader(
                        subset_dataset,
                        batch_size=batch_size,
                        num_workers=self.num_workers,
                        shuffle=self.shuffle_dataset,
                    )
                )
            else:
                subset_dataset = JoinedDataset(
                    dataset="train",
                    config=config,
                    download_dataset=self.download_dataset,
                    heatmap_kernel_size=heatmap_kernel_size,
                    drone_view_patch_size=drone_view_patch_size,
                    metadata_rtree_index=self.metadata_rtree_index,
                )
                self.train_dataloaders.append(
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
                    if self.train_subset_size is None
                    else subset_dataset.dataset.metadata_rtree_index
                )

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

    def train(self):
        """
        Train the model for a specified number of epochs.

        epochs: the number of epochs to train for
        """
        logger.info("Starting training...")
        for epoch in range(self.current_epoch, self.num_epochs):
            logger.info(f"Epoch {epoch + 1}")
            self.train_epoch()
            self.save_checkpoint(epoch=epoch)

            # reduce learning rate for the 10th and 14th epochs
            if epoch in [9, 13]:
                self.scheduler.step()

            # Validate every 2 epochs
            if (epoch + 1) % 2 == 0:
                logger.info("Validating...")
                self.validate()

    def train_epoch(self):
        """
        Perform one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        total_samples = 0
        for dataloader, h_kernel_size, d_patch_size in zip(
            self.train_dataloaders,
            self.heatmap_kernel_sizes,
            self.drone_view_patch_sizes,
        ):
            logger.info(
                f"Training epoch with kernel size {h_kernel_size} and patch size {d_patch_size}"
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

                # Zero out the gradients
                self.optimizer.zero_grad()
                # Forward pass
                outputs = self.model(drone_images, sat_images)
                # Calculate loss
                loss = self.criterion(outputs, heatmap_gt)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                if i == 0 and self.plot:
                    self.plot_results(
                        drone_images[0].detach(),
                        sat_images[0].detach(),
                        heatmap_gt[0].detach(),
                        outputs[0].detach(),
                    )

                running_loss += loss.item() * drone_images.size(0)
            total_samples += len(dataloader)

        epoch_loss = running_loss / total_samples
        logger.info(f"Training loss: {epoch_loss}")

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
                            drone_images[0], sat_images[0], heatmap_gt[0], outputs[0]
                        )
                total_samples += len(dataloader)

        epoch_loss = running_loss / total_samples
        logger.info(f"Validation loss: {epoch_loss}")

    def plot_results(
        self,
        drone_image,
        sat_image,
        heatmap_gt,
        heatmap_pred,
    ):
        """
        Plot the outputs of the model and the ground truth.
        """
        # Plot them on the same figure
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(1, 4, 1)
        ax.imshow(drone_image.permute(1, 2, 0).cpu().numpy())
        ax.set_title("Drone Image")
        ax.axis("off")

        ax = fig.add_subplot(1, 4, 2)
        ax.imshow(sat_image.permute(1, 2, 0).cpu().numpy())
        ax.set_title("Satellite Image")
        ax.axis("off")

        ax = fig.add_subplot(1, 4, 3)
        ax.imshow(heatmap_gt.squeeze(0).cpu().numpy(), cmap="viridis")
        ax.set_title("Ground Truth Heatmap")
        ax.axis("off")

        ax = fig.add_subplot(1, 4, 4)
        ax.imshow(heatmap_pred.squeeze(0).cpu().numpy(), cmap="viridis")
        ax.set_title("Predicted Heatmap")
        ax.axis("off")

        plt.show()

    def save_checkpoint(self, epoch, dir_path="./checkpoints/"):
        """
        Save the current state of the model to a checkpoint file.

        epoch: current epoch number
        dir_path: the directory to save the checkpoint to
        """
        os.makedirs(dir_path, exist_ok=True)
        train_dir = f"{dir_path}/{self.checkpoint_hash}/"
        os.makedirs(train_dir, exist_ok=True)
        save_path = f"{train_dir}/checkpoint-{epoch}.pt"

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            save_path,
        )

    def load_checkpoint(self, dir_path="./checkpoints/", epoch=0):
        """
        Load the model state from a checkpoint file.

        dir_path: the directory to load the checkpoint from
        epoch: the epoch to load the checkpoint from
        """
        checkpoint_path = f"{dir_path}/{self.checkpoint_hash}/checkpoint-{epoch}.pt"
        checkpoint = torch.load(checkpoint_path)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint["epoch"]


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Modified twins model for cross-view localization training script"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configuration",
        help="Path to the configuration file",
    )

    args = parser.parse_args()

    config = load_config(f"./conf/{args.config}.yaml")

    train_config = config["train"]

    device = torch.device(train_config["device"])
    loss_fn = torch.nn.MSELoss(reduction="mean")
    # loss_fn = WeightedMSELoss()

    trainer = CrossViewTrainer(
        device,
        loss_fn,
        config=config,
    )

    trainer.train()


if __name__ == "__main__":
    main()
