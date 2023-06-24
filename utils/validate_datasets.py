#!/usr/bin/env python3
import mercantile
import csv
import matplotlib.pyplot as plt
import os
import json
import multiprocessing as mp
from functools import partial

# Constants
tiles_path = "../sat/"
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


def find_on_map(lat, lng, metadata_reader, plot=False):
    for row in metadata_reader:
        tile = mercantile.Tile(int(row["x"]), int(row["y"]), int(row["z"]))

        if is_coord_in_a_tile(lat, lng, tile):
            x, y = coord_to_pixel(lat, lng, tile, SAT_DIM_IM, SAT_DIM_IM)

            # open the image
            fp = row["file_path"]
            if plot:
                im_map = plt.imread(tiles_path + fp)
                return im_map, tile
            else:
                return None, tile

    return None, None


def sort_by_last_digits(arr):
    def key_func(s):
        return int(s.split("_")[-1].replace(".jpeg", ""))

    return sorted(arr, key=key_func)


def validate_dataset(
    i_path="../drone/Train1_MB_150m_80fov_90deg/footage/",
    m_path="../drone/Train1_MB_150m_80fov_90deg/Train1_MB_150m_80fov_90deg.json",
    plot=False,
    metadata_reader=None,
):
    with open(m_path, newline="") as jsonfile:
        json_dict = json.load(jsonfile)

        camera_frames = json_dict["cameraFrames"]

        imgs = []
        for filename in os.listdir(i_path):
            if filename.endswith(".jpeg"):
                imgs.append(filename)
            else:
                continue

        imgs = sort_by_last_digits(imgs)

        for im, frame in zip(imgs, camera_frames):
            map_im, tile = find_on_map(
                frame["coordinate"]["latitude"],
                frame["coordinate"]["longitude"],
                metadata_reader,
            )

            if tile is None:
                raise Exception(
                    f"Could not find tile for {frame['coordinate']['latitude']}, {frame['coordinate']['longitude']}!\n"
                    f"[-] Validation failed..."
                )

            if plot and map_im is not None:
                fig = plt.figure(figsize=(10, 10))
                ax = fig.subplots(1, 2)

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


def validate_directory(directory, drone_dir, metadata_reader, plot=False):
    if os.path.isdir(drone_dir + directory):
        if "Train" in directory or "Test" in directory or "Val" in directory:
            print(f"Validating {directory}...")
            validate_dataset(
                f"{drone_dir}{directory}/footage/",
                f"{drone_dir}{directory}/{directory}.json",
                plot=plot,
                metadata_reader=metadata_reader,
            )
            print(f"Validation of {directory} successful!")


def iterate_through_datasets(metadata_reader, drone_dir="../drone/"):
    with mp.Pool(mp.cpu_count()) as pool:
        f = partial(
            validate_directory,
            drone_dir=drone_dir,
            metadata_reader=metadata_reader,
            plot=False,
        )
        pool.map(f, os.listdir(drone_dir))


def read_metadata_file(metadata_path="../sat/metadata.csv"):
    with open(metadata_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        return rows


def main():
    metadata_reader = read_metadata_file()
    iterate_through_datasets(metadata_reader)


if __name__ == "__main__":
    main()
