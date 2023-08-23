#!/usr/bin/env python3
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import functional as F
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import rasterio.warp
import re
import mercantile
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.merge import merge
import warnings
from rasterio.errors import NotGeoreferencedWarning
import time
import requests
import gc


class CastralDataset(Dataset):
    """
    Custom dataset class for the Drone Dataset.

    Args:
        dataset (str): Dataset type, either "train" or "test".
        config (dict): Configuration dictionary containing dataset parameters.
        patch_h (int): Height of the image patch.
        patch_w (int): Width of the image patch.

    """

    def __init__(
        self,
        dataset="train",
        config=None,
        drone_patch_w=None,
        drone_patch_h=None,
        sat_patch_w=None,
        sat_patch_h=None,
        heatmap_kernel_size=None,
        heatmap_type=None,
        tiffs=None,
    ):
        config = config["dataset"]
        self.root_dir = config["root_dir"]
        self.patch_w = drone_patch_w if drone_patch_w else config["drone_patch_w"]
        self.patch_h = drone_patch_h if drone_patch_h else config["drone_patch_h"]
        self.sat_patch_w = sat_patch_w if sat_patch_w else config["sat_patch_w"]
        self.sat_patch_h = sat_patch_h if sat_patch_h else config["sat_patch_h"]
        self.heatmap_kernel_size = (
            heatmap_kernel_size
            if heatmap_kernel_size
            else config["heatmap_kernel_size"]
        )
        self.heatmap_type = config["heatmap_type"]
        self.metadata_dict = {}
        self.dataset = dataset
        self.tiffs = tiffs if tiffs else config["tiffs"]
        self.image_paths = self.get_entry_paths(self.root_dir)
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=config["mean"], std=config["std"]),
            ]
        )
        if self.dataset == "train":
            self.random_seed = config.get("random_seed", 42)
            self.set_seed(self.random_seed)

        self.prepare_kernels()

        self.deterministic_val = True if dataset == "test" else False

        if self.deterministic_val:
            self.drone_to_sat_dict = {}

        self.headers = {
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

        self.params = {
            "access_token": "pk.eyJ1Ijoib3BlbnN0cmVldG1hcCIsImEiOiJjbGRlaGp1b3gwOGRtM250NW9sOHhuMmRjIn0.Y3mM21ciEP5Zo5amLJUugg",
        }

        self.tiles_path = "./sat"
        self.sat_zoom_level = 16

        # Suppress the NotGeoreferencedWarning warning
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = (
                True  # Slows down a bit, but ensures reproducibility
            )

    def get_entry_paths(self, path):
        """
        Recursively retrieves paths to image and metadata files in the given directory.

        Args:
            path (str): Path to the directory.

        Returns:
            List[str]: List of file paths.

        """
        entry_paths = []
        entries = os.listdir(path)
        for entry in entries:
            entry_path = path + "/" + entry
            if os.path.isdir(entry_path):
                entry_paths += self.get_entry_paths(entry_path + "/")
            if self.dataset == "train" and not ("YESlapo_STD" in entry_path):
                if entry_path.endswith(".tif"):
                    entry_paths.append(entry_path)
            elif self.dataset == "test" and "YESlapo_STD" in entry_path:
                if entry_path.endswith(".tif"):
                    entry_paths.append(entry_path)
        return entry_paths

    def prepare_kernels(self):
        """
        Sets the hanning kernel for heatmap generation.

        Args:
            kernel_size (int): Size of the kernel.

        """
        hann1d = torch.hann_window(self.heatmap_kernel_size, periodic=False)
        self.hanning_kernel = hann1d.unsqueeze(1) * hann1d.unsqueeze(0)

        gauss1d = torch.linspace(-1, 1, self.heatmap_kernel_size)
        gauss1d = torch.exp(-gauss1d.pow(2))
        self.gaussian_kernel = gauss1d.unsqueeze(1) * gauss1d.unsqueeze(0)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.

        """
        return len(self.image_paths)

    def haversine_np(self, lat1, lon1, lat2, lon2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371.0 * 1000 * c

    def geo_to_pixel_coordinates(self, lat, lon, transform):
        x_pixel, y_pixel = ~transform * (lon, lat)
        return round(x_pixel), round(y_pixel)

    def get_heatmap_gt(self, x, y, height, width, square_size=33):
        if self.heatmap_type == "hanning":
            return self.generate_hanning_heatmap(x, y, height, width, square_size)
        elif self.heatmap_type == "gaussian":
            return self.generate_gaussian_heatmap(x, y, height, width, square_size)
        elif self.heatmap_type == "square":
            return self.generate_square_heatmap(x, y, height, width, square_size)
        else:
            raise ValueError("Invalid heatmap type: ", self.heatmap_type)

    def download_missing_tile(self, tile):
        os.makedirs(f"{self.tiles_path}", exist_ok=True)
        file_path = f"{self.tiles_path}/tiles/{tile.z}_{tile.x}_{tile.y}.jpg"

        if os.path.exists(file_path):
            return

        max_attempts = 5
        for attempt in range(max_attempts):
            print(
                f"Downloading tile {tile.z}_{tile.x}_{tile.y} (attempt {attempt + 1}/{max_attempts})..."
            )
            try:
                response = requests.get(
                    f"https://c.tiles.mapbox.com/v4/mapbox.satellite/{tile.z}/{tile.x}/{tile.y}@2x.jpg",
                    params=self.params,
                    headers=self.headers,
                )
                response.raise_for_status()  # raises a Python exception if the response contains an HTTP error status code
            except (
                requests.exceptions.RequestException,
                requests.exceptions.ConnectionError,
            ) as e:
                if attempt < max_attempts - 1:  # i.e., if it's not the final attempt
                    print("Error downloading tile. Retrying...")
                    time.sleep(5)  # wait for 5 seconds before trying again
                    continue
                else:
                    print("Error downloading tile. Max retries exceeded.")
                    break
            else:  # executes if the try block didn't throw any exceptions
                with open(file_path, "wb") as f:
                    f.write(response.content)
                break
        else:
            print("Error downloading tile. Max retries exceeded.")

    def get_tiff_map(self, tile):
        tile_data = []
        neighbors = mercantile.neighbors(tile)
        neighbors.append(tile)

        for neighbor in neighbors:
            found = False
            west, south, east, north = mercantile.bounds(neighbor)
            tile_path = (
                f"{self.tiles_path}/tiles/{neighbor.z}_{neighbor.x}_{neighbor.y}.jpg"
            )
            if os.path.exists(tile_path):
                found = True
                with Image.open(tile_path) as img:
                    width, height = img.size

                memfile = MemoryFile()
                with memfile.open(
                    driver="GTiff",
                    height=height,
                    width=width,
                    count=3,
                    dtype="uint8",
                    crs="EPSG:3857",
                    transform=from_bounds(west, south, east, north, width, height),
                ) as dataset:
                    data = rasterio.open(tile_path).read()
                    dataset.write(data)
                tile_data.append(memfile.open())

                del memfile

            if not found:
                self.download_missing_tile(neighbor)
                time.sleep(1)
                tile_path = f"{self.tiles_path}/tiles/{neighbor.z}_{neighbor.x}_{neighbor.y}.jpg"
                with Image.open(tile_path) as img:
                    width, height = img.size
                memfile = MemoryFile()
                with memfile.open(
                    driver="GTiff",
                    height=height,
                    width=width,
                    count=3,
                    dtype="uint8",
                    crs="EPSG:3857",
                    transform=from_bounds(west, south, east, north, width, height),
                ) as dataset:
                    data = rasterio.open(tile_path).read()
                    dataset.write(data)
                tile_data.append(memfile.open())
                del memfile

        mosaic, out_trans = merge(tile_data)

        out_meta = tile_data[0].meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "crs": "EPSG:3857",
            }
        )

        # Clean up MemoryFile instances to free up memory
        for td in tile_data:
            td.close()

        del neighbors
        del tile_data
        gc.collect()

        return mosaic, out_meta

    def get_tile_from_coord(self, lat, lng, zoom_level):
        tile = mercantile.tile(lng, lat, zoom_level)
        return tile

    def get_random_tiff_patch(self, path, lat, lon, patch_width, patch_height):
        """
        Returns a random patch from the satellite image.
        """

        tile = self.get_tile_from_coord(lat, lon, self.sat_zoom_level)
        mosaic, out_meta = self.get_tiff_map(tile)

        transform = out_meta["transform"]

        x_pixel, y_pixel = self.geo_to_pixel_coordinates(lat, lon, transform)

        ks = self.heatmap_kernel_size // 2

        x_offset_range = [
            x_pixel - patch_width + ks + 1,
            x_pixel - ks - 1,
        ]
        y_offset_range = [
            y_pixel - patch_height + ks + 1,
            y_pixel - ks - 1,
        ]

        # Randomly select an offset within the valid range
        x_offset = random.randint(*x_offset_range)
        y_offset = random.randint(*y_offset_range)

        # Define the window based on the offsets and patch size
        window = Window(x_offset, y_offset, patch_width, patch_height)

        # Read the data within the window
        x, y = x_pixel - x_offset, y_pixel - y_offset
        patch = mosaic[:, y : y + patch_height, x : x + patch_width]

        return patch, x, y, x_offset, y_offset, transform

    def get_tiff_patch(self, lat, lon, patch_width, patch_height, x_offset, y_offset):
        """
        Returns a patch from the satellite image with the given offset and size.
        """

        tile = self.get_tile_from_coord(lat, lon, self.sat_zoom_level)
        mosaic, out_meta = self.get_tiff_map(tile)

        transform = out_meta["transform"]
        x_pixel, y_pixel = self.geo_to_pixel_coordinates(lat, lon, transform)

        # Validate the window dimensions and offsets
        if (
            x_offset < 0
            or x_offset + patch_width > mosaic.shape[2]
            or y_offset < 0
            or y_offset + patch_height > mosaic.shape[1]
        ):
            raise ValueError("Invalid patch parameters")

        # Extract the data from the mosaic based on the defined window
        patch = mosaic[
            :, y_offset : y_offset + patch_height, x_offset : x_offset + patch_width
        ]

        # Adjust pixel coordinates relative to the patch
        x, y = x_pixel - x_offset, y_pixel - y_offset

        return patch, x, y, x_offset, y_offset, transform

    def generate_square_heatmap(self, x, y, height, width, square_size=33):
        """
        Generates a square heatmap centered at the given coordinates.

        Args:
            lat (float): Latitude.
            lng (float): Longitude.
            sat_image (torch.Tensor): Satellite image tensor.
            x (int): x-coordinate of the tile.
            y (int): y-coordinate of the tile.
            z (int): Zoom level of the tile.
            square_size (int): Size of the square kernel.

        Returns:
            torch.Tensor: Heatmap tensor.

        """

        x_map, y_map = x, y

        heatmap = torch.zeros((height, width))

        half_size = square_size // 2

        # Calculate the valid range for the square
        start_x = max(0, x_map - half_size)
        end_x = min(
            width, x_map + half_size + 1
        )  # +1 to include the end_x in the square
        start_y = max(0, y_map - half_size)
        end_y = min(
            height, y_map + half_size + 1
        )  # +1 to include the end_y in the square

        heatmap[start_y:end_y, start_x:end_x] = 1

        return heatmap

    def generate_hanning_heatmap(self, x, y, height, width, square_size=33):
        x_map, y_map = x, y

        height, width = height, width

        heatmap = torch.zeros((height, width))

        # Compute half size of the hanning window
        half_size = self.hanning_kernel.shape[0] // 2

        # Calculate the valid range for the hanning window
        start_x = max(0, x_map - half_size)
        end_x = min(width, start_x + self.hanning_kernel.shape[1])
        start_y = max(0, y_map - half_size)
        end_y = min(height, start_y + self.hanning_kernel.shape[0])

        # If the hanning window doesn't fit at the current position, move its start position
        if end_x - start_x < self.hanning_kernel.shape[1]:
            start_x = end_x - self.hanning_kernel.shape[1]

        if end_y - start_y < self.hanning_kernel.shape[0]:
            start_y = end_y - self.hanning_kernel.shape[0]

        # Assign the hanning window to the valid region within the heatmap tensor
        heatmap[start_y:end_y, start_x:end_x] = self.hanning_kernel[
            : end_y - start_y, : end_x - start_x
        ]

        heatmap = heatmap / heatmap.sum()

        return heatmap

    def generate_gaussian_heatmap(self, x, y, height, width, square_size=33):
        """
        Generates a heatmap using a Gaussian kernel centered at the given coordinates.

        Args:
            lat (float): Latitude.
            lng (float): Longitude.
            sat_image (torch.Tensor): Satellite image tensor.
            x (int): x-coordinate of the tile.
            y (int): y-coordinate of the tile.
            z (int): Zoom level of the tile.
            square_size (int): Size of the square kernel.

        Returns:
            torch.Tensor: Heatmap tensor.

        """
        x_map, y_map = x, y

        heatmap = torch.zeros((width, height))

        # Compute half size of the Gaussian window
        half_size = self.gaussian_kernel.shape[0] // 2

        # Calculate the valid range for the Gaussian window
        start_x = max(0, x_map - half_size)
        end_x = min(width, start_x + self.gaussian_kernel.shape[1])
        start_y = max(0, y_map - half_size)
        end_y = min(height, start_y + self.gaussian_kernel.shape[0])

        # If the Gaussian window doesn't fit at the current position, move its start position
        if end_x - start_x < self.gaussian_kernel.shape[1]:
            start_x = end_x - self.gaussian_kernel.shape[1]

        if end_y - start_y < self.gaussian_kernel.shape[0]:
            start_y = end_y - self.gaussian_kernel.shape[0]

        # Assign the Gaussian window to the valid region within the heatmap tensor
        heatmap[start_y:end_y, start_x:end_x] = self.gaussian_kernel[
            : end_y - start_y, : end_x - start_x
        ]

        # Normalize the heatmap
        heatmap = heatmap / heatmap.sum()

        return heatmap

    def get_patch_center_latlon(self, transform, width, height, src_crs):
        """Compute the center of the image patch in latitude and longitude."""
        center_x = transform.c + (width / 2) * transform.a
        center_y = transform.f + (height / 2) * transform.e

        # Convert to lat and lon using rasterio's warp function
        lon, lat = rasterio.warp.transform(src_crs, "EPSG:4326", [center_x], [center_y])
        return lat[0], lon[0]

    def __getitem__(self, idx):
        """
        Retrieves and preprocesses the image and its corresponding metadata at the given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: Tuple containing the preprocessed image tensor and the metadata dictionary.

        """
        image_path = self.image_paths[idx]

        drone_im_tif = None
        img_info = {
            "filename": image_path,
        }

        with rasterio.open(image_path) as drone_tif:
            transform = drone_tif.transform
            src_crs = drone_tif.crs

            # Get the center of the image patch in latitude and longitude
            lat, lon = self.get_patch_center_latlon(
                transform, self.patch_w, self.patch_h, src_crs
            )

            drone_im_tif = drone_tif.read()

        if self.deterministic_val:
            key = str(idx)

            # Check if key is already in self.drone_to_sat_dict
            if key in self.drone_to_sat_dict:
                x_offset, y_offset = self.drone_to_sat_dict[key]
                (
                    satellite_patch,
                    x_sat,
                    y_sat,
                    x_offset,
                    y_offset,
                    sat_transform,
                ) = self.get_tiff_patch(
                    image_path,
                    lat,
                    lon,
                    self.sat_patch_h,
                    self.sat_patch_w,
                    x_offset,
                    y_offset,
                )
            else:
                (
                    satellite_patch,
                    x_sat,
                    y_sat,
                    x_offset,
                    y_offset,
                    sat_transform,
                ) = self.get_random_tiff_patch(
                    image_path, lat, lon, self.sat_patch_h, self.sat_patch_h
                )
                self.drone_to_sat_dict[key] = (x_offset, y_offset)

        else:
            (
                satellite_patch,
                x_sat,
                y_sat,
                x_offset,
                y_offset,
                sat_transform,
            ) = self.get_random_tiff_patch(
                image_path, lat, lon, self.sat_patch_h, self.sat_patch_w
            )

        img_info["scale"] = 1.0
        drone_im_tif = drone_im_tif.transpose(1, 2, 0)
        # Drop the alpha channel
        drone_im_tif = drone_im_tif[:, :, :3]
        drone_im_tif = Image.fromarray(drone_im_tif)
        drone_image = F.resize(drone_im_tif, (self.patch_h, self.patch_w))
        drone_image = self.transforms(drone_image)

        satellite_patch = satellite_patch.transpose(1, 2, 0)
        satellite_patch = self.transforms(satellite_patch)

        # Generate heatmap
        heatmap = self.get_heatmap_gt(
            x_sat,
            y_sat,
            satellite_patch.shape[1],
            satellite_patch.shape[2],
            self.heatmap_kernel_size,
        )

        img_info["x_sat"] = x_sat
        img_info["y_sat"] = y_sat
        img_info["x_offset"] = x_offset
        img_info["y_offset"] = y_offset
        img_info["zoom_level"] = 16
        img_info["lat"] = lat
        img_info["lon"] = lon
        img_info["sat_transform"] = sat_transform

        return drone_image, img_info, satellite_patch, heatmap

    def extract_info_from_filename(self, filename):
        """
        Extracts information from the filename.

        Args:
            filename (str): Filename of the image.

        Returns:
            tuple: Tuple containing the extracted information and the file number.

        """
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


def test():
    import yaml

    with open("./conf/castral.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset = CastralDataset(config=config)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch in dataloader:
        drone_images, drone_infos, satellite_images, heatmaps = batch

        print("Drone images shape: ", drone_images.shape)
        print("Satellite images shape: ", satellite_images.shape)
        assert drone_images.shape == (len(batch[0]), 3, 128, 128)
        assert satellite_images.shape == (len(batch[0]), 3, 400, 400)
        print(drone_infos)

        # First pair of drone and satellite images
        fig, axs = plt.subplots(1, 3, figsize=(20, 6))
        axs[0].imshow(drone_images[0].permute(1, 2, 0))
        axs[0].set_title("Drone image 1")
        axs[1].imshow(satellite_images[0].permute(1, 2, 0))
        axs[1].set_title("Corresponding Satellite image 1")
        # Scatter the drone's location on the satellite Image
        axs[1].scatter(drone_infos["x_sat"][0], drone_infos["y_sat"][0], c="r", s=40)
        # Display heatmap
        axs[2].imshow(heatmaps[0], cmap="hot", interpolation="nearest")
        axs[2].set_title("Heatmap 1")
        plt.show()

        # Second pair of drone and satellite images
        fig, axs = plt.subplots(1, 3, figsize=(20, 6))
        axs[0].imshow(drone_images[1].permute(1, 2, 0))
        axs[0].set_title("Drone image 2")
        axs[1].imshow(satellite_images[1].permute(1, 2, 0))
        axs[1].set_title("Corresponding Satellite image 2")
        # Scatter the drone's location on the satellite image
        axs[1].scatter(drone_infos["x_sat"][1], drone_infos["y_sat"][1], c="r", s=40)
        # Display heatmap
        axs[2].imshow(heatmaps[1], cmap="hot", interpolation="nearest")
        axs[2].set_title("Heatmap 2")
        plt.show()

    print("Test successful!")


if __name__ == "__main__":
    test()
