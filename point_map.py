#!/usr/bin/env python3
import mercantile
import csv
import matplotlib.pyplot as plt
import os
import json

# Constants
tiles_path = "./sat/"
SAT_DIM_IM = 512


def pixel_to_coord(x, y, tile, image_width, image_height):
    bbox = mercantile.bounds(tile)
    min_lng, min_lat, max_lng, max_lat = bbox
    pixel_width = (max_lng - min_lng) / image_width
    pixel_height = (max_lat - min_lat) / image_height

    lng = min_lng + x * pixel_width
    lat = max_lat - y * pixel_height

    return lat, lng


def coord_to_pixel(lat, lng, tile, image_width, image_height):
    bbox = mercantile.bounds(tile)
    min_lng, min_lat, max_lng, max_lat = bbox
    pixel_width = (max_lng - min_lng) / image_width
    pixel_height = (max_lat - min_lat) / image_height

    x = (lng - min_lng) / pixel_width
    y = (max_lat - lat) / pixel_height

    return x, y


def is_coord_in_a_tile(lat, lng, tile):
    bbox = mercantile.bounds(tile)
    min_lng, min_lat, max_lng, max_lat = bbox

    return lat >= min_lat and lat <= max_lat and lng >= min_lng and lng <= max_lng


# Kongresni trg: 46.050246, 14.503744
def find_on_map(lat, lng):
    with open(tiles_path + "metadata.csv", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            tile = mercantile.Tile(int(row["x"]), int(row["y"]), int(row["z"]))

            if is_coord_in_a_tile(lat, lng, tile):
                print("found lat lan: " + str(tile))
                x, y = coord_to_pixel(lat, lng, tile, SAT_DIM_IM, SAT_DIM_IM)

                # open the image
                fp = row["file_path"]
                im_map = plt.imread(tiles_path + fp)
                return im_map, tile


def sort_by_last_digits(arr):
    # Define a function to be used as the sorting key.
    # This function takes a string, splits it on the underscore character,
    # takes the last element, removes the ".jpeg" extension, and converts the result to an integer.
    def key_func(s):
        return int(s.split("_")[-1].replace(".jpeg", ""))

    # Use the sorted() function with the custom key function to sort the array.
    return sorted(arr, key=key_func)


def read_maribor(
    i_path="./drone/Train1_MB_150m_80fov_90deg/footage/",
    m_path="./drone/Train1_MB_150m_80fov_90deg/Train1_MB_150m_80fov_90deg.json",
):
    with open(m_path, newline="") as jsonfile:
        json_dict = json.load(jsonfile)

        camera_frames = json_dict["cameraFrames"]

        # List all the image files in the i_path directory
        imgs = []
        for filename in os.listdir(i_path):
            if filename.endswith(".jpeg"):
                imgs.append(filename)
            else:
                continue

        imgs = sort_by_last_digits(imgs)

        for im, frame in zip(imgs, camera_frames):
            print(frame)
            map_im, tile = find_on_map(
                frame["coordinate"]["latitude"], frame["coordinate"]["longitude"]
            )
            if map_im is not None:
                # Create matplotlib figure
                fig = plt.figure(figsize=(10, 10))
                ax = fig.subplots(1, 2)

                # Read the imageA
                im_drone = plt.imread(i_path + im)

                # Plot the map and drone image
                ax[0].imshow(map_im)
                ax[1].imshow(im_drone)

                # Plot the drone position
                x, y = coord_to_pixel(
                    frame["coordinate"]["latitude"],
                    frame["coordinate"]["longitude"],
                    tile,
                    SAT_DIM_IM,
                    SAT_DIM_IM,
                )
                ax[0].scatter(x, y, c="r", s=50)

                plt.show()


read_maribor()
