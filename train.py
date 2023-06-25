#!/usr/bin/env python3
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from model import CustomResNetDeiT
from criterion import BalanceLoss
from joined_dataset import JoinedDataset
from torch.utils.data import DataLoader


class CrossViewTrainer:
    """Trainer class for cross-view (UAV and satellite) image learning"""

    def __init__(
        self, backbone, dataloader, device, criterion, lr=3e-4, weight_decay=5e-4
    ):
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
        self.dataloader = dataloader
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
        for epoch in range(epochs):
            self.train_epoch()
            self.save_checkpoint()

            # reduce learning rate for the 10th and 14th epochs
            if epoch in [9, 13]:
                self.scheduler.step()

    def train_epoch(self):
        """
        Perform one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        for i, (drone_images, drone_infos, sat_images, sat_infos) in tqdm(
            enumerate(self.dataloader)
        ):
            drone_images = drone_images.to(self.device)
            sat_images = sat_images.to(self.device)

            # Zero out the gradients
            self.optimizer.zero_grad()
            # Forward pass
            outputs = self.model(drone_images, sat_images)
            # Calculate loss
            loss = self.criterion(
                outputs, outputs
            )  # TODO: implement ground truth labels
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * drone_images.size(0)

        epoch_loss = running_loss / len(self.dataloader.dataset)
        print("Training Loss: {:.4f}".format(epoch_loss))

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
    model = CustomResNetDeiT()
    model = torch.nn.DataParallel(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(
        JoinedDataset(drone_dir="./drone/", sat_dir="./sat"),
        batch_size=8,
        shuffle=True,
        num_workers=16,
    )
    loss_fn = BalanceLoss()

    trainer = CrossViewTrainer(model, dataloader, device, loss_fn)

    trainer.train(epochs=15)


if __name__ == "__main__":
    test()
