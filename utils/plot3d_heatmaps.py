#!/usr/bin/env python

import os
import csv
import math
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def parse_hann_kers_csv(file_path):
    """
    Parse hann_kers.csv to get a mapping of hash to kernel size.
    """
    hash_to_kernel = {}
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        next(reader)  # skip header
        for row in reader:
            size, hash_value = row
            hash_to_kernel[hash_value.strip()] = size
    return hash_to_kernel


def get_kernel_size_from_file(file, hash_to_kernel):
    """
    Extract kernel size from the file name using the hash_to_kernel mapping.
    """
    hash_value = file.split("_")[-1]
    hash_value = hash_value.split("-")[0].strip()
    return int(hash_to_kernel.get(hash_value, 0))


def get_rds_from_file(file_name, directory):
    """
    Extract RDS value from the JSON file corresponding to the given image file name.
    """
    json_file_name = file_name.replace("_3d_hm_", "_").replace(".png", ".json")
    json_path = os.path.join(directory, json_file_name)

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            return data["rds"]
    except (FileNotFoundError, KeyError):
        return None


def plot_images_from_directory(directory, hash_to_kernel):
    """
    Plot images from a directory with Hann kernel size and RDS as title.
    """
    files = [
        f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    ]

    files = [f for f in files if not f.endswith(".json")]

    # Sort files by kernel size
    files = sorted(files, key=lambda f: get_kernel_size_from_file(f, hash_to_kernel))

    plt.figure(figsize=(10, 25))  # Adjust the figure size to make it compact

    for idx, file in enumerate(files):
        hash_value = file.split("_")[-1]
        hash_value = hash_value.split("-")[0].strip()
        kernel_size = hash_to_kernel.get(hash_value, "Unknown")

        rds_value = get_rds_from_file(file, directory)
        title = f"Velikost hanningovega okna: {kernel_size}"

        if rds_value is not None:
            title += f", RDS: {rds_value:.2f}"

        img_path = os.path.join(directory, file)
        img = mpimg.imread(img_path)

        ax = plt.subplot(5, 2, idx + 1)  # 5 rows, 2 columns for 10 images
        ax.imshow(img)
        ax.axis("off")  # Hide axes
        ax.set_title(title, fontsize=14)

    plt.tight_layout(pad=0)  # Remove padding between subplots
    plt.subplots_adjust(wspace=0, hspace=0)  # Remove whitespace between subplots
    plt.savefig("res/combined_heatmaps3d.png")


if __name__ == "__main__":
    hash_to_kernel = parse_hann_kers_csv("res/hann_kers.csv")
    plot_images_from_directory("res/heatmaps3d", hash_to_kernel)
