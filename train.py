#!/usr/bin/env python3
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from criterion import DiceLoss
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


class CrossViewTrainer:
    """Trainer class for cross-view (UAV and satellite) image learning"""

    def __init__(
        self,
        device,
        criterion,
        lr=3e-4,
        weight_decay=5e-4,
        batch_size=2,
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
        self.device = device
        self.criterion = criterion
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.shuffle_dataset = shuffle_dataset
        self.checkpoint_hash = checkpoint_hash
        self.checkpoint_epoch = checkpoint_epoch
        self.train_subset_size = train_subset_size
        self.val_subset_size = val_subset_size
        self.plot = plot
        self.current_epoch = 0
        self.download_dataset = config["train"]["download_dataset"]

        if self.train_subset_size is not None:
            logger.info(f"Using train subset of size {self.train_subset_size}")
            subset_dataset = torch.utils.data.Subset(
                JoinedDataset(
                    dataset="train",
                    config=config,
                    download_dataset=self.download_dataset,
                ),
                indices=range(self.train_subset_size),
            )
            self.train_dataloader = DataLoader(
                subset_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle_dataset,
            )
        else:
            subset_dataset = JoinedDataset(
                dataset="train",
                config=config,
                download_dataset=self.download_dataset,
            )
            self.train_dataloader = DataLoader(
                subset_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle_dataset,
            )

        if self.val_subset_size is not None:
            logger.info(f"Using val subset of size {self.val_subset_size}")
            subset_dataset = torch.utils.data.Subset(
                JoinedDataset(
                    dataset="test",
                    config=config,
                    download_dataset=self.download_dataset,
                ),
                indices=range(self.val_subset_size),
            )
            self.val_dataloader = DataLoader(
                subset_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle_dataset,
            )
        else:
            subset_dataset = JoinedDataset(
                dataset="test", config=config, download_dataset=self.download_dataset
            )
            self.val_dataloader = DataLoader(
                subset_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle_dataset,
            )

        self.model = torch.nn.DataParallel(
            CrossViewLocalizationModel(
                drone_resolution=self.train_dataloader.dataset.drone_resolution
                if self.train_subset_size is None
                else self.train_dataloader.dataset.dataset.drone_resolution,
                satellite_resolution=self.train_dataloader.dataset.satellite_resolution
                if self.train_subset_size is None
                else self.train_dataloader.dataset.dataset.satellite_resolution,
            )
        )

        self.optimizer = AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.1)

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
            if epoch % 2 == 0:
                logger.info("Validating...")
                self.validate()

    def train_epoch(self):
        """
        Perform one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        for i, (drone_images, drone_infos, sat_images, sat_infos, heatmap_gt) in tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
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

            running_loss += loss.item() * drone_images.size(0)

        epoch_loss = running_loss / len(self.train_dataloader)
        logger.info("Training Loss: {:.4f}".format(epoch_loss))

    def validate(self):
        """
        Perform one epoch of validation.
        """
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i, (
                drone_images,
                drone_infos,
                sat_images,
                sat_infos,
                heatmap_gt,
            ) in tqdm(
                enumerate(self.val_dataloader),
                total=len(self.val_dataloader),
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
                    self.plot_res(outputs, heatmap_gt, drone_images, sat_images)

        epoch_loss = running_loss / len(self.val_dataloader)
        logger.info("Validation Loss: {:.4f}".format(epoch_loss))

    def plot_res(self, outputs, heatmap_gt, drone_images, sat_images):
        """
        Plot the outputs of the model and the ground truth.
        """
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(outputs[0].cpu().squeeze().numpy(), cmap="hot")
        ax[0].set_title("Predicted Heatmap")
        ax[1].imshow(heatmap_gt[0].cpu().squeeze().numpy(), cmap="hot")
        ax[1].set_title("Ground Truth Heatmap")
        plt.show()

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(drone_images[0].permute(1, 2, 0).cpu().squeeze().numpy())
        ax[0].set_title("Drone Image")
        ax[1].imshow(sat_images[0].cpu().squeeze().numpy())
        ax[1].set_title("Satellite Image")
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
    loss_fn = DiceLoss()

    trainer = CrossViewTrainer(
        device,
        loss_fn,
        batch_size=train_config["batch_size"],
        num_workers=train_config["num_workers"],
        shuffle_dataset=train_config["shuffle_dataset"],
        num_epochs=train_config["num_epochs"],
        val_subset_size=train_config["val_subset_size"],
        train_subset_size=train_config["train_subset_size"],
        plot=train_config["plot"],
        checkpoint_hash=train_config["checkpoint_hash"],
        checkpoint_epoch=train_config["checkpoint_epoch"],
        config=config,
    )

    trainer.train()


if __name__ == "__main__":
    main()
