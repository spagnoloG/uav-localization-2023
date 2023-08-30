#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import mercantile
from matplotlib.colors import LinearSegmentedColormap
from functools import lru_cache
import torch


class MapUtils:
    """
    Utility class for satellite map operations.

    """

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

    def distance_between_points(self, lat1, lon1, lat2, lon2):
        """
        Calculates the distance between two points on the map.

        Args:
            lat1 (float): Latitude of the first point.
            lon1 (float): Longitude of the first point.
            lat2 (float): Latitude of the second point.
            lon2 (float): Longitude of the second point.

        Returns:
            float: Distance between the two points.

        """
        # Convert coordinates from degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Earth radius in meters
        R = 6371000

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # Haversine formula
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c

        return distance

    def pixel_to_coord(self, x, y, tile, image_width, image_height):
        """
        Converts pixel coordinates to latitude and longitude coordinates.

        Args:
            x (int): x-coordinate.
            y (int): y-coordinate.
            tile (mercantile.Tile): Tile object.
            image_width (int): Width of the image.
            image_height (int): Height of the image.

        Returns:
            tuple: Tuple containing the latitude and longitude coordinates.

        """
        bbox = mercantile.bounds(tile)
        min_lng, min_lat, max_lng, max_lat = bbox
        pixel_width = (max_lng - min_lng) / image_width
        pixel_height = (max_lat - min_lat) / image_height

        lng = min_lng + x * pixel_width
        lat = max_lat - y * pixel_height

        return lat, lng

    def coord_to_pixel(self, lat, lng, tile, image_width, image_height):
        """
        Converts latitude and longitude coordinates to pixel coordinates.

        Args:
            lat (float): Latitude.
            lng (float): Longitude.
            tile (mercantile.Tile): Tile object.
            image_width (int): Width of the image.
            image_height (int): Height of the image.

        Returns:
            tuple: Tuple containing the x and y pixel coordinates.

        """
        bbox = mercantile.bounds(tile)
        min_lng, min_lat, max_lng, max_lat = bbox
        pixel_width = (max_lng - min_lng) / image_width
        pixel_height = (max_lat - min_lat) / image_height

        x = (lng - min_lng) / pixel_width
        y = (max_lat - lat) / pixel_height

        return x, y

    def is_coord_in_a_tile(self, lat, lng, tile):
        """
        Checks if the given coordinates are within the bounds of the given tile.

        Args:
            lat (float): Latitude.
            lng (float): Longitude.
            tile (mercantile.Tile): Tile object.

        Returns:
            bool: True if the coordinates are within the tile, False otherwise.

        """
        bbox = mercantile.bounds(tile)
        min_lng, min_lat, max_lng, max_lat = bbox

        return lat >= min_lat and lat <= max_lat and lng >= min_lng and lng <= max_lng

    def find_and_plot(self, lat, lng, tile, map_image):
        """
        Finds the pixel coordinates of the given latitude and longitude coordinates in the satellite map
        and plots the map with a marker at the corresponding location.

        Args:
            lat (float): Latitude.
            lng (float): Longitude.
            tile (mercantile.Tile): Tile object.
            map_image (numpy.ndarray): Satellite map image.

        """
        x, y = self.coord_to_pixel(
            lat, lng, tile, map_image.shape[1], map_image.shape[2]
        )
        x, y = int(x), int(y)

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.imshow(map_image.permute(1, 2, 0))
        ax.scatter(x, y, color="red")
        plt.show()

    def plot_heatmap(self, lat, lng, sat_image, x, y, z):
        """
        Plots a heatmap centered at the given coordinates on top of the satellite image.

        Args:
            lat (float): Latitude.
            lng (float): Longitude.
            sat_image (numpy.ndarray): Satellite image.
            x (int): x-coordinate of the tile.
            y (int): y-coordinate of the tile.
            z (int): Zoom level of the tile.

        """
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
        """
        Creates a 2D Gaussian distribution.

        Args:
            dim1 (int): Dimension 1 of the output array.
            dim2 (int): Dimension 2 of the output array.
            sigma (float): Standard deviation of the Gaussian distribution.
            center (tuple): Center coordinates of the Gaussian distribution. If None, the center is set to
                            (dim1 // 2, dim2 // 2).

        Returns:
            numpy.ndarray: 2D Gaussian array.

        """
        if center is None:
            center = [dim1 // 2, dim2 // 2]
        x = np.arange(0, dim1, 1, dtype=np.float32)
        y = np.arange(0, dim2, 1, dtype=np.float32)
        x, y = np.meshgrid(x, y)
        return np.exp(
            -4 * np.log(2) * ((x - center[0]) ** 2 + (y - center[1]) ** 2) / sigma**2
        )

    def RDS(self, k, dx, dy, hm_w, hm_h):
        """
        Computes the Relative Distance Score (RDS) for given distances and heatmap dimensions.

        Args:
            k (float): Decay coefficient.
            dx, dy (float): Distances in x and y directions, respectively.
            hm_w, hm_h (int): Width and height of the heatmap.

        Returns:
            float: Relative Distance Score.
        """
        return np.exp(-k * np.sqrt(((dx / hm_w) ** 2 + (dy / hm_h) ** 2)) / 2)

    def MA(self, x_pred, y_pred, x_gt, y_gt, k=10):
        """
        Calculates the Metre Level Accuracy (MA) between predicted and ground truth positions.

        Args:
            x_pred, y_pred (float): Predicted coordinates.
            x_gt, y_gt (float): Ground truth coordinates.
            k (int, optional): Unused parameter (for potential future use). Default is 10.

        Returns:
            float: Metre Level Accuracy between predicted and ground truth.
        """
        dx = abs(x_pred.item() - x_gt)
        dy = abs(y_pred.item() - y_gt)
        return (dx**2 + dy**2) ** 0.5
