#!/usr/bin/env python3
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from model import CustomResNetDeiT
from criterion import BalanceLoss
from joined_dataset import JoinedDataset
from torch.utils.data import DataLoader
from logger import logger


class CrossViewTrainer:
    """Trainer class for cross-view (UAV and satellite) image learning"""

    def __init__(self, backbone, device, criterion, lr=3e-4, weight_decay=5e-4):
        """
        Initialize the CrossViewTrainer.

        backbone: the pretrained DeiT-S model, with its classifier removed
        dataloader: the dataloaders for the UAV and satellite view images, respectively
        device: the device to train on
        criterion: the loss function to use
        lr: learning rate
        weight_decay: weight decay for the optimizer
        """
        self.model = backbone
        self.train_dataloader = DataLoader(
            JoinedDataset(
                dataset="train",
            )
        )
        self.val_dataloader = DataLoader(
            JoinedDataset(
                dataset="test",
            )
        )
        self.device = device
        self.criterion = criterion

        self.model.to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.1)

    def train(self, epochs):
        """
        Train the model for a specified number of epochs.

        epochs: the number of epochs to train for
        """
        logger.info("Starting training...")
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}")
            self.train_epoch()
            self.save_checkpoint()

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

        epoch_loss = running_loss / len(self.dataloader.dataset)
        logger.log("Training Loss: {:.4f}".format(epoch_loss))

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

        epoch_loss = running_loss / len(self.val_dataloader.dataset)
        logger.log("Validation Loss: {:.4f}".format(epoch_loss))

    def save_checkpoint(self):
        """
        Save the current state of the model to a checkpoint file.
        """
        # TODO

    def load_checkpoint(self, checkpoint_path):
        """
        Load the model state from a checkpoint file.

        checkpoint_path: the path to the checkpoint file to load
        """
        # TODO


def test():
    model = CustomResNetDeiT(train_backbone=True, train_convolutions=True)
    model = torch.nn.DataParallel(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fn = BalanceLoss()

    trainer = CrossViewTrainer(model, device, loss_fn)

    trainer.train(epochs=15)


if __name__ == "__main__":
    test()
