#!/usr/bin/env python3
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import torch


class DroneDataset(Dataset):
    def __init__(self, root_dir="./drone/", patch_w=224, patch_h=224):
        self.root_dir = root_dir
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.metadata_dict = {}
        self.image_paths = self.get_entry_paths(self.root_dir)

    def get_entry_paths(self, path):
        entry_paths = []
        entries = os.listdir(path)
        for entry in entries:
            entry_path = path + "/" + entry
            if os.path.isdir(entry_path):
                entry_paths += self.get_entry_paths(entry_path + "/")
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
        return len(self.image_paths) * 18

    def __getitem__(self, idx):
        image_path = self.image_paths[idx // 18]
        image = Image.open(image_path)

        lookup_str, file_number = self.extract_info_from_filename(image_path)
        img_info = self.metadata_dict[lookup_str][file_number]
        img_info["filename"] = image_path

        rotation_angle = (idx % 18) * 20
        img_info["angle"] = rotation_angle

        image = image.rotate(rotation_angle)
        image = self.crop_center(image, self.patch_w, self.patch_h)

        image = np.array(image, dtype=np.float32)
        image = image / 255.0
        image = torch.from_numpy(image)  # Convert to PyTorch tensor

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

    def crop_center(self, img, crop_width, crop_height):
        img_width, img_height = img.size
        return img.crop(
            (
                (img_width - crop_width) // 2,
                (img_height - crop_height) // 2,
                (img_width + crop_width) // 2,
                (img_height + crop_height) // 2,
            )
        )


def test():
    dataloader = torch.utils.data.DataLoader(
        DroneDataset(root_dir="./drone/"), batch_size=10, shuffle=True
    )

    for batch_idx, (images, infos) in enumerate(dataloader):
        fig, axs = plt.subplots(2, 5, figsize=(15, 6))
        axs = axs.ravel()

        for i, (image, angle) in enumerate(zip(images, infos["angle"])):
            axs[i].imshow(image)
            axs[i].set_title(f"Image {i+1}: {angle.item()}")
            axs[i].axis("off")

        plt.show()


if __name__ == "__main__":
    test()
