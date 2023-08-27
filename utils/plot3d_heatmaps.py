#!/usr/bin/env python

import os
import csv
import math
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
            print(row)
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


def plot_images_from_directory(directory, hash_to_kernel):
    """
    Plot images from a directory with Hann kernel size as title.
    """
    files = [
        f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    ]

    # Sort files by kernel size
    files = sorted(files, key=lambda f: get_kernel_size_from_file(f, hash_to_kernel))

    n_cols = math.ceil(len(files) / 2)

    plt.figure(figsize=(10 * n_cols, 20))

    for idx, file in enumerate(files):
        hash_value = file.split("_")[-1]
        hash_value = hash_value.split("-")[0].strip()
        kernel_size = hash_to_kernel.get(hash_value, "Unknown")

        img_path = os.path.join(directory, file)
        img = mpimg.imread(img_path)

        ax = plt.subplot(2, n_cols, idx + 1)
        ax.imshow(img)
        ax.axis("off")  # Hide axes
        ax.set_title(f"Velikost hanningovega okna: {kernel_size}")

    plt.tight_layout()
    plt.savefig("res/heatmaps3d.png")


if __name__ == "__main__":
    hash_to_kernel = parse_hann_kers_csv("res/hann_kers.csv")
    plot_images_from_directory("res/heatmaps3d", hash_to_kernel)
