#!/usr/bin/env python3
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import functional as F


class DroneDataset(Dataset):
    def __init__(
        self,
        dataset="train",
        config=None,
        patch_h=None,
        patch_w=None,
    ):
        self.root_dir = config["root_dir"]
        self.patch_w = patch_w if patch_w else config["patch_w"]
        self.patch_h = patch_h if patch_h else config["patch_h"]
        self.metadata_dict = {}
        self.dataset = dataset
        self.rotation_deg = config["rotation_deg"]
        self.rotations_per_image = (
            360 // self.rotation_deg
        )  # Number of rotations for each image
        self.image_paths = self.get_entry_paths(self.root_dir)
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=config["mean"], std=config["std"]),
            ]
        )

    def get_entry_paths(self, path):
        entry_paths = []
        entries = os.listdir(path)
        for entry in entries:
            entry_path = path + "/" + entry
            if os.path.isdir(entry_path):
                entry_paths += self.get_entry_paths(entry_path + "/")
            if self.dataset == "train" and "Train" in entry_path:
                if entry_path.endswith(".jpeg"):
                    entry_paths.append(entry_path)
                if entry_path.endswith(".json"):
                    self.get_metadata(entry_path)
            elif self.dataset == "test" and "Test" in entry_path:
                if entry_path.endswith(".jpeg"):
                    entry_paths.append(entry_path)
                if entry_path.endswith(".json"):
                    self.get_metadata(entry_path)
        return entry_paths

    def get_metadata(self, path):
        with open(path, newline="") as jsonfile:
            json_dict = json.load(jsonfile)
            path = path.split("/")[-1]
            path = path.replace(".json", "")
            self.metadata_dict[path] = json_dict["cameraFrames"]

    def __len__(self):
        return len(self.image_paths) * self.rotations_per_image

    def __getitem__(self, idx):
        image_path = self.image_paths[idx // self.rotations_per_image]
        image = Image.open(image_path).convert("RGB")  # Ensure 3-channel image

        lookup_str, file_number = self.extract_info_from_filename(image_path)
        img_info = self.metadata_dict[lookup_str][file_number]
        img_info["filename"] = image_path

        rotation_angle = (idx % self.rotations_per_image) * self.rotation_deg
        img_info["angle"] = rotation_angle

        # Rotate crop center and transform image
        image = F.rotate(image, rotation_angle)
        image = F.center_crop(image, (self.patch_h, self.patch_w))
        image = self.transforms(image)

        return image, img_info

    def extract_info_from_filename(self, filename):
        filename_without_ext = filename.replace(".jpeg", "")
        segments = filename_without_ext.split("/")
        info = segments[-1]
        try:
            number = int(info.split("_")[-1])
        except ValueError:
            print("Could not extract number from filename: ", filename)
            return None, None

        info = "_".join(info.split("_")[:-1])

        return info, number


def calculate_mean_and_std(dataset):
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count()
    )

    mean = 0.0
    std = 0.0
    nb_samples = 0.0
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean.tolist(), std.tolist()


def test():
    import yaml

    with open("./conf/configuration.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = config["drone_dataset"]

    dataset = DroneDataset(config=config)
    # print(calculate_mean_and_std(dataset))

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    for batch_idx, (images, infos) in enumerate(dataloader):
        fig, axs = plt.subplots(2, 5, figsize=(15, 6))
        axs = axs.ravel()

        for i, (image, angle) in enumerate(zip(images, infos["angle"])):
            if i >= 10:
                break
            axs[i].imshow(image.permute(1, 2, 0).numpy())
            axs[i].set_title(f"Image {i+1}: {angle.item()}")
            axs[i].axis("off")

        plt.show()


if __name__ == "__main__":
    test()
