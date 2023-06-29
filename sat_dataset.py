#!/usr/bin/env python3
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import mercantile
from tqdm import tqdm
import requests
import time
from sat.bounding_boxes import bboxes
from matplotlib.colors import LinearSegmentedColormap
from logger import logger
import torch.nn.functional as F
from functools import lru_cache
import rtree


class SatDataset(Dataset):
    # Original file size -> 512 x 512
    def __init__(self, config, download_dataset=False):
        self.root_dir = config["root_dir"]
        self.patch_w = config["patch_w"]
        self.patch_h = config["patch_h"]
        self.zoom_level = config["zoom_level"]
        self.metadata_dict = {}
        self.image_indices = {}
        self.image_paths = self.get_entry_paths(self.root_dir)
        self.download_dataset = download_dataset
        self.rtree_index = rtree.index.Index()

        if self.download_dataset:
            self.download_maps()

        self.fill_metadata_dict()

    def fill_metadata_dict(self):
        logger.info("Building rtree index...")
        for idx, image_path in enumerate(self.image_paths):
            img_info = self.extract_info_from_filename(image_path)
            self.metadata_dict[image_path] = img_info
            self.image_indices[image_path] = idx
            self.rtree_index.insert(idx, mercantile.bounds(img_info))
        logger.info("Finished building rtree index.")

    def get_entry_paths(self, path):
        entry_paths = []
        entries = os.listdir(path)
        for entry in entries:
            entry_path = path + "/" + entry
            if os.path.isdir(entry_path):
                entry_paths += self.get_entry_paths(entry_path + "/")
            if entry_path.endswith(".jpg"):
                entry_paths.append(entry_path)
        return entry_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        metadata = self.metadata_dict[image_path]
        image = Image.open(image_path)
        image = self.downsample(image, self.patch_w, self.patch_h)

        image = np.array(image, dtype=np.float32)
        image = image / 255.0
        image = torch.from_numpy(image)

        return image, (metadata.x, metadata.y, metadata.z)  # return as integers

    def downsample(self, image, new_width, new_height):
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        return resized_image

    def extract_info_from_filename(self, filename):
        # 16_35582_23023.jpg -> z, x, y
        fn = filename.split("/")[-1]
        fn = fn.split(".")[0]
        fn = fn.split("_")
        try:
            z, x, y = int(fn[0]), int(fn[1]), int(fn[2])
            if z < 1:
                raise ValueError
        except ValueError:
            logger.error(
                "Invalid zoom level or error extracting info from the filename in sat images: ",
                filename,
            )
            return None

        try:
            tile = mercantile.Tile(x, y, z)
        except ValueError:
            logger.error(
                "Invalid tile or error extracting info from the filename in sat images: ",
                filename,
            )
            return None

        return tile

        ### -------------------------- ###
        ### MAP MANIPULATION FUNCTIONS ###
        ### -------------------------- ###

    def find_tile(self, lat, lng):
        possible_indices = list(self.rtree_index.intersection((lng, lat, lng, lat)))
        for idx in possible_indices:
            path = self.image_paths[idx]
            tile = self.metadata_dict[path]
            if self.is_coord_in_a_tile(lat, lng, tile):
                return self.__getitem__(idx)
        logger.error("No tile found for the given coordinates: ", lat, lng)
        raise ValueError("No tile found for the given coordinates: ", lat, lng)

    def is_coord_in_a_tile(self, lat, lng, tile):
        bbox = mercantile.bounds(tile)
        min_lng, min_lat, max_lng, max_lat = bbox

        return lat >= min_lat and lat <= max_lat and lng >= min_lng and lng <= max_lng

    ### -------------------------- ###
    ### DOWNLOADING MAPS FUNCTIONS ###
    ### -------------------------- ###

    def download_maps(self):
        headers = {
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Referer": "https://www.openstreetmap.org/",
            "Sec-Fetch-Dest": "image",
            "Sec-Fetch-Mode": "no-cors",
            "Sec-Fetch-Site": "cross-site",
            "Sec-GPC": "1",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "sec-ch-ua": '"Not.A/Brand";v="8", "Chromium";v="114", "Brave";v="114"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
        }

        params = {
            "access_token": "pk.eyJ1Ijoib3BlbnN0cmVldG1hcCIsImEiOiJjbGRlaGp1b3gwOGRtM250NW9sOHhuMmRjIn0.Y3mM21ciEP5Zo5amLJUugg",
        }

        os.makedirs(f"{self.root_dir}/tiles", exist_ok=True)

        for r_name, bbox in bboxes.items():
            logger.info(f"Downloading maps for region: {r_name}")
            for tile in tqdm(
                mercantile.tiles(bbox[0], bbox[1], bbox[2], bbox[3], self.zoom_level)
            ):
                box = mercantile.bounds(tile)

                center_lat = (box.north + box.south) / 2
                center_lng = (box.east + box.west) / 2

                os.makedirs(f"{self.root_dir}/tiles/{r_name}", exist_ok=True)
                file_path = f"{self.root_dir}/tiles/{r_name}/{self.zoom_level}_{tile.x}_{tile.y}.jpg"

                if not os.path.exists(file_path):
                    max_attempts = 5
                    for attempt in range(max_attempts):
                        try:
                            response = requests.get(
                                f"https://c.tiles.mapbox.com/v4/mapbox.satellite/{self.zoom_level}/{tile.x}/{tile.y}@2x.jpg",
                                params=params,
                                headers=headers,
                            )
                            response.raise_for_status()  # raises a Python exception if the response contains an HTTP error status code
                        except (
                            requests.exceptions.RequestException,
                            requests.exceptions.ConnectionError,
                        ) as e:
                            if (
                                attempt < max_attempts - 1
                            ):  # i.e., if it's not the final attempt
                                logger.warn(
                                    f"An error occurred: {e}. Trying again in 5 seconds..."
                                )
                                time.sleep(5)  # wait for 5 seconds before trying again
                                continue
                            else:
                                logger.error(
                                    f"Failed to download after {max_attempts} attempts. Skipping this tile."
                                )
                                break
                        else:  # executes if the try block didn't throw any exceptions
                            with open(file_path, "wb") as f:
                                f.write(response.content)
                            break
                    else:
                        logger.error(f"Error downloading {file_path}")


class MapUtils:
    def __init__(self):
        self.heatmap_colors = [
            "darkblue",
            "blue",
            "lightblue",
            "green",
            "lightgreen",
            "yellow",
            "gold",
            "orange",
            "darkorange",
            "red",
            "darkred",
        ]
        self.Center_R = 33
        self.prepare_hanning_window()

    def prepare_hanning_window(self):
        hann1d = torch.hann_window(self.Center_R)
        self.hanning_window = hann1d.unsqueeze(1) * hann1d.unsqueeze(0)

    def pixel_to_coord(self, x, y, tile, image_width, image_height):
        bbox = mercantile.bounds(tile)
        min_lng, min_lat, max_lng, max_lat = bbox
        pixel_width = (max_lng - min_lng) / image_width
        pixel_height = (max_lat - min_lat) / image_height

        lng = min_lng + x * pixel_width
        lat = max_lat - y * pixel_height

        return lat, lng

    def coord_to_pixel(self, lat, lng, tile, image_width, image_height):
        bbox = mercantile.bounds(tile)
        min_lng, min_lat, max_lng, max_lat = bbox
        pixel_width = (max_lng - min_lng) / image_width
        pixel_height = (max_lat - min_lat) / image_height

        x = (lng - min_lng) / pixel_width
        y = (max_lat - lat) / pixel_height

        return x, y

    def is_coord_in_a_tile(self, lat, lng, tile):
        bbox = mercantile.bounds(tile)
        min_lng, min_lat, max_lng, max_lat = bbox

        return lat >= min_lat and lat <= max_lat and lng >= min_lng and lng <= max_lng

    def find_and_plot(self, lat, lng, tile, map_image):
        print("lat, lng, tile", lat, lng, tile)
        x, y = self.coord_to_pixel(
            lat, lng, tile, map_image.shape[0], map_image.shape[1]
        )
        x, y = int(x), int(y)

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.imshow(map_image)
        ax.scatter(x, y, color="red")
        plt.show()

    def generate_heatmap(self, lat, lng, sat_image, x, y, z, square_size=33):
        tile = mercantile.Tile(x=x, y=y, z=z)
        x_map, y_map = self.coord_to_pixel(
            lat, lng, tile, sat_image.shape[0], sat_image.shape[1]
        )

        x_map, y_map = int(x_map), int(y_map)

        height, width = sat_image.shape[0], sat_image.shape[1]

        heatmap = torch.zeros((height, width))

        # Compute half size of the hanning window
        half_size = self.hanning_window.shape[0] // 2

        # Calculate the valid range for the hanning window
        start_x = max(0, x_map - half_size)
        end_x = min(width, start_x + self.hanning_window.shape[1])
        start_y = max(0, y_map - half_size)
        end_y = min(height, start_y + self.hanning_window.shape[0])

        # If the hanning window doesn't fit at the current position, move its start position
        if end_x - start_x < self.hanning_window.shape[1]:
            start_x = end_x - self.hanning_window.shape[1]

        if end_y - start_y < self.hanning_window.shape[0]:
            start_y = end_y - self.hanning_window.shape[0]

        # Assign the hanning window to the valid region within the heatmap tensor
        heatmap[start_y:end_y, start_x:end_x] = self.hanning_window[
            : end_y - start_y, : end_x - start_x
        ]

        return heatmap

    def plot_heatmap(self, lat, lng, sat_image, x, y, z):
        tile = mercantile.Tile(x=x, y=y, z=z)
        x_map, y_map = self.coord_to_pixel(
            lat, lng, tile, sat_image.shape[0], sat_image.shape[1]
        )
        x_map, y_map = int(x_map), int(y_map)

        std_dev = 20
        heatmap = self.create_2d_gaussian(
            sat_image.shape[1], sat_image.shape[0], std_dev, center=[x_map, y_map]
        )
        heatmap /= np.max(heatmap)
        fig, ax = plt.subplots()
        ax.imshow(sat_image)
        cmap = LinearSegmentedColormap.from_list("heatmap", self.heatmap_colors)
        ax.imshow(heatmap, cmap=cmap, alpha=0.5)
        plt.show()

    @lru_cache(maxsize=None)
    def create_2d_gaussian(self, dim1, dim2, sigma=1, center=None):
        if center is None:
            center = [dim1 // 2, dim2 // 2]
        x = np.arange(0, dim1, 1, dtype=np.float32)
        y = np.arange(0, dim2, 1, dtype=np.float32)
        x, y = np.meshgrid(x, y)
        return np.exp(
            -4 * np.log(2) * ((x - center[0]) ** 2 + (y - center[1]) ** 2) / sigma**2
        )


def test():
    import yaml

    with open("./conf/configuration.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = config["sat_dataset"]

    dataloader = torch.utils.data.DataLoader(
        SatDataset(config=config), batch_size=10, shuffle=True
    )
    map_utils = MapUtils()

    # for batch_idx, (images, infos) in enumerate(dataloader):
    #    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    #    axs = axs.ravel()

    #    print(infos)

    #    for i in range(len(images)):
    #        image = images[i]
    #        info = infos[0][i]
    #        axs[i].imshow(image)
    #        print(f"Info for image {i}: {info}")
    #        axs[i].axis("off")

    #    plt.show()

    img, metadata = dataloader.dataset.find_tile(46.051450, 14.506099)
    tile = mercantile.Tile(x=metadata[0], y=metadata[1], z=metadata[2])

    map_utils.find_and_plot(46.051450, 14.506099, tile, img)


if __name__ == "__main__":
    test()
