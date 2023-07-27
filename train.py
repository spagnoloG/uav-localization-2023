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
import matplotlib.pyplot as plt
from model import CrossViewLocalizationModel
import yaml
import argparse
import torchvision.transforms as transforms
from criterion import HanningLoss, RDS
import os
import numpy as np
from map_utils import MapUtils
import matplotlib.patches as patches
import json
import itertools

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class ConvergenceEarlyStopping:
    """Early stopping to stop the training when the loss does not improve after
    certain epochs, and reduce the learning rate when the loss does not improve
    for a specified number of consecutive epochs.
    """

    def __init__(self, scheduler, patience=3):
        """
        :param scheduler: the scheduler
        :param patience: how many epochs to wait before stopping the training when loss is not improving
        """
        self.scheduler = scheduler
        self.best_loss = None
        self.stale_epochs = 0
        self.stale_epochs_reseted = False
        self.patience = patience

    def step(self, val_loss):
        """
        Update the learning rate and check if we need to early stop the training.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss <= val_loss:
            self.stale_epochs += 1
            logger.warning(f"Loss has not improved for {self.stale_epochs} epochs")
            if self.stale_epochs == self.patience:
                if self.stale_epochs_reseted:
                    logger.warning("Loss has not improved after reducing learning rate")
                    return True

                logger.warning(
                    f"Loss has not improved for {self.stale_epochs} epochs, reducing learning rate"
                )
                self.scheduler.step()
                self.stale_epochs = 0
                self.stale_epochs_reseted = True
        else:
            self.stale_epochs = 0
            prev_best_loss = self.best_loss
            self.best_loss = val_loss
            self.stale_epochs_reseted = False
            logger.info(f"Loss has improved from {prev_best_loss} to {val_loss}")

        return False


class CrossViewTrainer:
    """Trainer class for cross-view (UAV and satellite) image learning"""

    def __init__(
        self,
        config=None,
    ):
        """
        Initialize the CrossViewTrainer.

        backbone: the pretrained DeiT-S model, with its classifier removed
        device: the device to train on
        criterion: the loss function to use
        lr: learning rate
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
        self.config = config
        self.device = config["train"]["device"]
        self.lr_fusion = config["train"]["lr_fusion"]
        self.lr_backbone = config["train"]["lr_backbone"]
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
        self.batch_size = config["train"]["batch_size"]
        self.train_until_convergence = config["train"]["train_until_convergence"]
        self.gamma = config["train"]["gamma"]
        self.val_loss = 0
        self.stop_training = False
        self.map_utils = MapUtils()
        self.RDS = RDS(k=10)
        self.heatmap_kernel_size = config["dataset"]["heatmap_kernel_size"]
        self.best_RDS = -np.inf

        if "cuda" in self.device:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = True

        self.criterion = HanningLoss(
            kernel_size=self.heatmap_kernel_size, device=self.device
        )

        if self.device == "cpu":
            self.model = CrossViewLocalizationModel(
                satellite_resolution=(
                    config["dataset"]["sat_patch_w"],
                    config["dataset"]["sat_patch_h"],
                )
            ).to(self.device)

            self.params_to_update_backbone = list(
                self.model.feature_extractor_UAV.parameters()
            ) + list(self.model.feature_extractor_satellite.parameters())
            self.params_to_update_fusion = list(self.model.fusion.parameters())
        else:
            self.model = torch.nn.DataParallel(  # support for multi-GPU training
                CrossViewLocalizationModel(
                    satellite_resolution=(
                        config["dataset"]["sat_patch_w"],
                        config["dataset"]["sat_patch_h"],
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
            self.optimizer, milestones=self.milestones, gamma=self.gamma
        )

        self.model.to(self.device)

        if self.checkpoint_hash is not None and self.checkpoint_epoch is not None:
            logger.info("Loading checkpoint...")
            try:
                self.current_epoch = self.load_checkpoint()
                self.current_epoch += 1  # Train from next epoch
            except FileNotFoundError:
                logger.error(
                    f"Checkpoint with hash {self.checkpoint_hash} not found. Starting from scratch."
                )

        else:
            logger.info("No checkpoint specified. Starting from scratch.")
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d-%H-%M-%S")
            now_hash = hashlib.sha1(now_str.encode()).hexdigest()
            self.checkpoint_hash = now_hash

        if self.train_until_convergence:
            self.convergence_early_stopping = ConvergenceEarlyStopping(
                scheduler=self.scheduler
            )

        logger.info("Preparing dataloaders...")
        self.prepare_dataloaders(config)
        logger.info("Dataloaders ready.")

        self.dump_config()

        logger.info(
            f"Using chekpoint hash {self.checkpoint_hash}, starting from epoch {self.current_epoch}"
        )

    def prepare_dataloaders(self, config):
        if self.train_subset_size is not None:
            logger.info(f"Using train subset of size {self.train_subset_size}")
            subset_dataset = torch.utils.data.Subset(
                JoinedDataset(
                    dataset="train",
                    config=config,
                ),
                indices=range(self.train_subset_size),
            )
            self.train_dataloader = DataLoader(
                subset_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=self.shuffle_dataset,
            )
        else:
            logger.info("Using full train dataset")
            subset_dataset = JoinedDataset(
                dataset="train",
                config=config,
            )
            self.train_dataloader = DataLoader(
                subset_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=self.shuffle_dataset,
            )

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

    def train(self):
        """
        Train the model for a specified number of epochs.

        epochs: the number of epochs to train for
        """
        logger.info("Starting training...")
        for epoch in range(self.current_epoch, self.num_epochs):
            logger.info(f"Epoch {epoch}")
            self.train_epoch(epoch)
            self.validate(epoch)

            if (epoch + 1) % 2 == 0:
                logger.info("Saving checkpoint...")
                self.save_checkpoint(epoch)

            if not self.train_until_convergence and epoch in self.milestones:
                logger.info("Stepping scheduler...")
                self.scheduler.step()

            if self.train_until_convergence and epoch > 10:
                stop_training = self.convergence_early_stopping.step(self.val_loss)
                self.stop_training = stop_training
                if stop_training:
                    break

        if not self.stop_training and self.train_until_convergence:
            while True:
                logger.info(f"Epoch {epoch}")
                self.train_epoch(epoch)
                self.validate(epoch)

                if (epoch + 1) % 2 == 0:
                    logger.info("Saving checkpoint...")
                    self.save_checkpoint(epoch)

                if not self.train_until_convergence and self.epoch in self.milestones:
                    self.scheduler.step()

                if self.train_until_convergence and epoch > 4:
                    stop_training = self.convergence_early_stopping.step(self.val_loss)

                if stop_training:
                    break

        return self.best_RDS

    def train_epoch(self, epoch):
        """
        Perform one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        running_RDS = 0.0
        for i, (
            drone_images,
            drone_infos,
            sat_images,
            heatmaps_gt,
            x_sat,
            y_sat,
        ) in tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
        ):
            drone_images = drone_images.to(self.device)
            sat_images = sat_images.to(self.device)
            heatmap_gt = heatmaps_gt.to(self.device)

            # Zero out the gradients
            self.optimizer.zero_grad()
            # Forward pass
            outputs = self.model(drone_images, sat_images)
            # Calculate loss
            loss = self.criterion(outputs, heatmap_gt)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            ### RDS ###
            with torch.no_grad():
                running_RDS += self.RDS(
                    outputs,
                    x_sat,
                    y_sat,
                    heatmaps_gt[0].shape[-1],
                    heatmaps_gt[0].shape[-2],
                ).item()
            ### RDS ###

            if i == 0 and self.plot:
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
                    f"train-{epoch}",
                )

            running_loss += loss.item() * drone_images.size(0)

        epoch_loss = running_loss / len(self.train_dataloader)
        logger.info(f"Training loss: {epoch_loss}")
        logger.info(f"Training RDS: {running_RDS / len(self.train_dataloader)}")

    def validate(self, epoch):
        """
        Perform one epoch of validation.
        """
        self.model.eval()
        running_loss = 0.0
        running_RDS = 0.0
        with torch.no_grad():
            for i, (
                drone_images,
                drone_infos,
                sat_images,
                heatmaps_gt,
                x_sat,
                y_sat,
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

                ### RDS ###
                running_RDS += self.RDS(
                    outputs,
                    x_sat,
                    y_sat,
                    heatmaps_gt[0].shape[-1],
                    heatmaps_gt[0].shape[-2],
                ).item()
                ### RDS ###

                if i == 0 and self.plot:
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
                        f"val-{epoch}-{i}",
                    )

        epoch_loss = running_loss / len(self.val_dataloader)
        self.val_loss = epoch_loss
        logger.info(f"Validation loss: {epoch_loss}")
        rds = running_RDS / len(self.val_dataloader)
        logger.info(f"Validation RDS: {rds}")
        self.best_RDS = max(self.best_RDS, rds)

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
        call_f,
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

        # Initialize figure
        fig = plt.figure(figsize=(20, 20))

        # Subplot 1: UAV Image
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.imshow(inverse_transforms(drone_image))
        ax1.set_title("UAV Image")
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
        ax6.axis("off")

        if "val" in call_f:
            s_dir = "val"
        else:
            s_dir = "train"

        # Save the figure
        os.makedirs(f"./vis/{self.checkpoint_hash}/{s_dir}", exist_ok=True)
        plt.savefig(
            f"./vis/{self.checkpoint_hash}/{s_dir}/{call_f}-{self.checkpoint_hash}-{i}.png"
        )
        plt.close()

    def dump_config(self, dir_path="./checkpoints/"):
        os.makedirs(f"{dir_path}/{self.checkpoint_hash}/", exist_ok=True)
        with open(f"{dir_path}/{self.checkpoint_hash}/config.json", "w") as f:
            json.dump(self.config, f)

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
        checkpoint_path = (
            f"{dir_path}/{self.checkpoint_hash}/checkpoint-{self.checkpoint_epoch}.pt"
        )
        checkpoint = torch.load(checkpoint_path)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint["epoch"]


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def hyperparameter_search(config):
    lr_backbone = [0.0001, 0.00001, 0.000001]
    lr_fusion = [0.0004, 0.0003, 0.0002, 0.0001]
    batch_size = [24]
    gamma = [0.1, 0.2, 0.3, 0.4, 0.5]
    milestones = [
        [9, 13, 15],
        [3, 5, 7],
        [8, 10, 11],
    ]

    all_params = list(
        itertools.product(lr_backbone, lr_fusion, batch_size, gamma, milestones)
    )
    train_subset_size = 18000
    val_subset_size = 1800
    best_score = float("-inf")
    best_params = None

    for params in all_params:
        print(f"Training with params: {params}")
        config["train"]["lr_backbone"] = params[0]
        config["train"]["lr_fusion"] = params[1]
        config["train"]["batch_size"] = params[2]
        config["train"]["gamma"] = params[3]
        config["train"]["milestones"] = params[4]
        config["train"]["train_subset_size"] = train_subset_size
        config["train"]["val_subset_size"] = val_subset_size
        trainer = CrossViewTrainer(config=config)
        train_score = trainer.train()

        if train_score < best_score:
            best_score = train_score
            best_params = params

    print(f"Best params: {best_params} with performance: {best_score}")


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

    hyperparameter_search(config)

    # trainer = CrossViewTrainer(
    #    config=config,
    # )

    # trainer.train()


if __name__ == "__main__":
    main()
