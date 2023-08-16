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

    def get_corresponding_tiff_sat(
        self, path, lat, lon, altitude, fov_vertical_deg, fov_horizontal_deg
    ):
        """ "
        Returns the same patch from the satellite image as the given patch from the drone image.
        """
        mm = re.search(r"(.*\/+[^\/]+)\/+[^\/]+\.tif", path)
        if mm:
            path = mm.group(1) + "/sat.tiff"
        else:
            raise ValueError("no bueno")

        with rasterio.open(path) as tif:
            transform = tif.transform

            x_res = abs(transform[0])
            y_res = abs(transform[4])

            lat2 = lat + y_res / 2
            lon2 = lon + x_res / 2

            y_res_meters = self.haversine_np(lat, lon, lat2, lon)
            x_res_meters = self.haversine_np(lat, lon, lat, lon2)

            fov_vertical_rad = np.radians(fov_vertical_deg)
            fov_horizontal_rad = np.radians(fov_horizontal_deg)

            height = np.tan(fov_vertical_rad / 2) * altitude
            width = np.tan(fov_horizontal_rad / 2) * altitude

            fov_height_pixels = int(height / y_res_meters)
            fov_width_pixels = int(width / x_res_meters)

            x_pixel, y_pixel = self.geo_to_pixel_coordinates(lat, lon, transform)

            window = Window(
                int(x_pixel - fov_width_pixels // 2),
                int(y_pixel - fov_height_pixels // 2),
                fov_width_pixels,
                fov_height_pixels,
            )

            if (
                window.width < 0
                or window.height < 0
                or window.col_off < 0
                or window.row_off < 0
            ):
                raise ValueError("Invalid window: ", window)

            data = tif.read(window=window)

            plt.imshow(data.transpose((1, 2, 0)))
            plt.show()

        return data

    def get_heatmap_gt(self, x, y, height, width, square_size=33):
        if self.heatmap_type == "hanning":
            return self.generate_hanning_heatmap(x, y, height, width, square_size)
        elif self.heatmap_type == "gaussian":
            return self.generate_gaussian_heatmap(x, y, height, width, square_size)
        elif self.heatmap_type == "square":
            return self.generate_square_heatmap(x, y, height, width, square_size)
        else:
            raise ValueError("Invalid heatmap type: ", self.heatmap_type)

    def get_random_tiff_patch(self, path, lat, lon, patch_width, patch_height):
        """ "
        Returns a random patch from the satellite image.
        """
        mm = re.search(r"(.*\/+[^\/]+)\/+[^\/]+\.tif", path)
        if mm:
            path = mm.group(1) + "/sat.tiff"
        else:
            raise ValueError("no bueno")

        with rasterio.open(path) as tif:
            transform = tif.transform

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
            patch = tif.read(window=window)

        return patch, x, y, x_offset, y_offset

    def get_tiff_patch(
        self, path, lat, lon, patch_width, patch_height, x_offset, y_offset
    ):
        """
        Returns a patch from the satellite image. with the given offset and size.
        """

        mm = re.search(r"(.*\/+[^\/]+)\/+[^\/]+\.tif", path)
        if mm:
            path = mm.group(1) + "/sat.tiff"
        else:
            raise ValueError("no bueno")

        with rasterio.open(path) as tif:
            transform = tif.transform
            x_pixel, y_pixel = self.geo_to_pixel_coordinates(lat, lon, transform)
            # Define the window based on the offsets and patch size
            window = Window(x_offset, y_offset, patch_width, patch_height)
            # Read the data within the window
            x, y = x_pixel - x_offset, y_pixel - y_offset
            patch = tif.read(window=window)

        return patch, x, y, x_offset, y_offset

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
        drone_image = Image.open(image_path).convert("RGB")  # Ensure 3-channel image

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
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in dataloader:
        drone_images, drone_infos, satellite_images, heatmaps = batch

        print("Drone images shape: ", drone_images.shape)
        print("Satellite images shape: ", satellite_images.shape)
        assert drone_images.shape == (len(batch[0]), 3, 128, 128)
        assert satellite_images.shape == (len(batch[0]), 3, 400, 400)

        # First pair of drone and satellite images
        fig, axs = plt.subplots(1, 3, figsize=(20, 6))
        axs[0].imshow(drone_images[0].permute(1, 2, 0))
        axs[0].set_title("Drone image 1")
        axs[1].imshow(satellite_images[0].permute(1, 2, 0))
        axs[1].set_title("Corresponding Satellite image 1")
        # Scatter the drone's location on the satellite Image
        # axs[1].scatter(x[0], y[0], c="r", s=40)
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
        # axs[1].scatter(x[1], y[1], c="r", s=40)
        # Display heatmap
        axs[2].imshow(heatmaps[1], cmap="hot", interpolation="nearest")
        axs[2].set_title("Heatmap 2")
        plt.show()

    print("Test successful!")


if __name__ == "__main__":
    test()
