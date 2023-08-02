#!/usr/bin/env python3
import mercantile
import csv
import matplotlib.pyplot as plt
import os
import json
from multiprocessing import Pool
from functools import partial
import requests
import time
from random import shuffle
from tqdm import tqdm
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.merge import merge
from PIL import Image


# Constants
tiles_path = "../sat/"
SAT_DIM_IM = 512
ZOOM_LEVEL = 18

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


def download_missing_tile(tile):
    os.makedirs(f"{tiles_path}", exist_ok=True)
    file_path = f"{tiles_path}/tiles/{tile.z}_{tile.x}_{tile.y}.jpg"

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
                params=params,
                headers=headers,
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

    return lat > min_lat and lat < max_lat and lng > min_lng and lng < max_lng


def get_tile_from_coord(lat, lng, zoom_level):
    tile = mercantile.tile(lng, lat, zoom_level)
    return tile


def join_tifs(tile, i_path):
    tile_data = []
    neighbors = mercantile.neighbors(tile)
    neighbors.append(tile)

    for neighbor in neighbors:
        found = False
        west, south, east, north = mercantile.bounds(neighbor)
        tile_path = f"{tiles_path}/tiles/{neighbor.z}_{neighbor.x}_{neighbor.y}.jpg"
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

        if not found:
            download_missing_tile(neighbor)
            time.sleep(1)
            tile_path = f"{tiles_path}/tiles/{neighbor.z}_{neighbor.x}_{neighbor.y}.jpg"
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

    i_path = i_path + f"_sat_{ZOOM_LEVEL}.tiff"
    with rasterio.open(i_path, "w", **out_meta) as dest:
        dest.write(mosaic)
        print(f"Saved {i_path} successfully!")

    for t in tile_data:  # Close all datasets
        t.close()


def find_on_map(i_path, lat, lng, plot=False):
    tile = get_tile_from_coord(lat, lng, ZOOM_LEVEL)
    x, y = coord_to_pixel(lat, lng, tile, SAT_DIM_IM, SAT_DIM_IM)
    assert x >= 0 and x <= SAT_DIM_IM
    assert y >= 0 and y <= SAT_DIM_IM
    join_tifs(tile, i_path)
    # open the image
    # fp = row["file_path"]
    # if plot:
    #    im_map = plt.imread(tiles_path + fp)
    #    return im_map, tile
    # else:
    #    return None, tile
    return None, tile


def sort_by_last_digits(arr):
    def key_func(s):
        return int(s.split("_")[-1].replace(".jpeg", ""))

    return sorted(arr, key=key_func)


def process_image_frame(data, i_path, plot):
    im, frame = data

    map_im, tile = find_on_map(
        f"{i_path}{im}",
        frame["coordinate"]["latitude"],
        frame["coordinate"]["longitude"],
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


def validate_dataset(
    i_path="../drone/Train1_MB_150m_80fov_90deg/footage/",
    m_path="../drone/Train1_MB_150m_80fov_90deg/Train1_MB_150m_80fov_90deg.json",
    plot=False,
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
        pairs = list(zip(imgs, camera_frames))
        shuffle(pairs)

        # The pool size can be adjusted to your needs
        with Pool(os.cpu_count()) as pool:
            f = partial(
                process_image_frame,
                i_path=i_path,
                plot=plot,
            )
            pool.map(f, pairs)


def validate_directory(directory, drone_dir, plot=False):
    if os.path.isdir(drone_dir + directory):
        if "Train" in directory or "Test" in directory or "Val" in directory:
            validate_dataset(
                f"{drone_dir}{directory}/footage/",
                f"{drone_dir}{directory}/{directory}.json",
                plot=plot,
            )
            print(f"Validation of {directory} successful!")


def iterate_through_datasets(drone_dir="../dataset/"):
    for directory in tqdm(os.listdir(drone_dir)):
        print(f"Downloading {directory}...")
        validate_directory(
            directory,
            drone_dir=drone_dir,
            plot=False,
        )


def main():
    iterate_through_datasets()


if __name__ == "__main__":
    main()
