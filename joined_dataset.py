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


class JoinedDataset(Dataset):
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
        self.drone_scales = config["drone_scales"]
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

    def get_metadata(self, path):
        """
        Extracts metadata from a JSON file and stores it in the metadata dictionary.

        Args:
            path (str): Path to the JSON file.

        """
        with open(path, newline="") as jsonfile:
            json_dict = json.load(jsonfile)
            path = path.split("/")[-1]
            path = path.replace(".json", "")
            self.metadata_dict[path] = json_dict["cameraFrames"]

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.

        """
        return len(self.image_paths) * len(self.drone_scales)

    def equirectangular_np(self, lat1, lon1, lat2, lon2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        x = dlon * np.cos((lat1 + lat2) / 2)
        y = dlat
        radius_earth = 6371.0
        return np.sqrt(x * x + y * y) * radius_earth * 1000

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
        path = path + "_sat.tiff"

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
        tiff_num = random.choice(self.tiffs)
        path = path + f"_sat_{tiff_num}.tiff"

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

        return patch, x, y, x_offset, y_offset, tiff_num

    def get_tiff_patch(
        self, path, lat, lon, patch_width, patch_height, x_offset, y_offset, tiff_num
    ):
        """
        Returns a patch from the satellite image. with the given offset and size.
        """

        path = path + f"_sat_{tiff_num}.tiff"

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

    def __getitem__(self, idx):
        """
        Retrieves and preprocesses the image and its corresponding metadata at the given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: Tuple containing the preprocessed image tensor and the metadata dictionary.

        """
        image_path = self.image_paths[idx // len(self.drone_scales)]
        drone_image = Image.open(image_path).convert("RGB")  # Ensure 3-channel image

        lookup_str, file_number = self.extract_info_from_filename(image_path)
        img_info = self.metadata_dict[lookup_str][file_number]
        img_info["filename"] = image_path

        lat, lon = (
            img_info["coordinate"]["latitude"],
            img_info["coordinate"]["longitude"],
        )

        if self.deterministic_val:
            key = str(idx)

            # Check if key is already in self.drone_to_sat_dict
            if key in self.drone_to_sat_dict:
                x_offset, y_offset, tiff_num = self.drone_to_sat_dict[key]
                satellite_patch, x_sat, y_sat, _, _, _ = self.get_tiff_patch(
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
                    tiff_num,
                ) = self.get_random_tiff_patch(
                    image_path, lat, lon, self.sat_patch_h, self.sat_patch_h
                )
                self.drone_to_sat_dict[key] = (x_offset, y_offset, tiff_num)

        else:
            satellite_patch, x_sat, y_sat, _, _, _ = self.get_random_tiff_patch(
                image_path, lat, lon, self.sat_patch_h, self.sat_patch_w
            )

        s_c_a_l_e = self.drone_scales[(idx % len(self.drone_scales))]
        img_info["scale"] = s_c_a_l_e

        # Rotate crop center and transform image
        h = np.ceil(drone_image.height // s_c_a_l_e).astype(int)
        w = np.ceil(drone_image.width // s_c_a_l_e).astype(int)

        drone_image = F.resize(drone_image, [h, w])
        drone_image = F.center_crop(drone_image, (self.patch_h, self.patch_w))
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

        return drone_image, img_info, satellite_patch, heatmap, x_sat, y_sat

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

    with open("./conf/configuration.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset = JoinedDataset(config=config)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # satellite_patch, x_sat, y_sat, x_offset, y_offset = dataset.get_random_tiff_patch(
    #    "./dataset/Test1_Ljubljana_150m_80fov_90deg/footage/Test1_Ljubljana_150m_80fov_90deg_0515.jpeg",
    #    46.050095,
    #    14.502725,
    #    400,
    #    400,
    # )

    ## Plot the satellite patch and the point on the satellite patch
    # fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    # axs[0].imshow(satellite_patch.transpose(1, 2, 0))
    # axs[0].scatter(x_sat, y_sat, c="r")
    # axs[0].set_title("Satellite patch")
    # axs[1].imshow(satellite_patch.transpose(1, 2, 0))
    # axs[1].scatter(x_sat + x_offset, y_sat + y_offset, c="r")
    # axs[1].set_title("Satellite patch with offset")
    # plt.show()

    for batch in dataloader:
        drone_images, drone_infos, satellite_images, heatmaps, x, y = batch
        assert drone_images.shape == (len(batch[0]), 3, 128, 128)
        assert satellite_images.shape == (len(batch[0]), 3, 400, 400)
        # First pair of drone and satellite images
        fig, axs = plt.subplots(1, 3, figsize=(20, 6))
        axs[0].imshow(drone_images[0].permute(1, 2, 0))
        axs[0].set_title("Drone image 1")
        axs[1].imshow(satellite_images[0].permute(1, 2, 0))
        axs[1].set_title("Corresponding Satellite image 1")
        # Scatter the drone's location on the satellite Image
        axs[1].scatter(x[0], y[0], c="r", s=40)
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
        axs[1].scatter(x[1], y[1], c="r", s=40)
        # Display heatmap
        axs[2].imshow(heatmaps[1], cmap="hot", interpolation="nearest")
        axs[2].set_title("Heatmap 2")
        plt.show()

    print("Test successful!")


if __name__ == "__main__":
    test()
