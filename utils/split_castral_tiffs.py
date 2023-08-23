# !/usr/bin/env python3

import rasterio
from rasterio.windows import Window
import os
import mercantile
import requests
import time
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.merge import merge
from pyproj import Transformer
import rasterio.warp
from multiprocessing import Pool
import gc


tiles_path = "../sat/"
SAT_DIM_IM = 512
ZOOM_LEVEL = 16

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


def get_tile_from_coord(lat, lng, zoom_level):
    tile = mercantile.tile(lng, lat, zoom_level)
    return tile


def get_patch_center_latlon(transform, width, height, src_crs):
    """Compute the center of the image patch in latitude and longitude."""
    center_x = transform.c + (width / 2) * transform.a
    center_y = transform.f + (height / 2) * transform.e

    # Convert to lat and lon using rasterio's warp function
    lon, lat = rasterio.warp.transform(src_crs, "EPSG:4326", [center_x], [center_y])
    return lat[0], lon[0]


def split_tiff_rasterio(input_file, output_folder, tile_size_x, tile_size_y):
    with rasterio.open(input_file) as src:
        # Print the shape of the raster
        height, width = src.shape
        transform = src.transform

        n_tiles_x = int((width + tile_size_x - 1) / tile_size_x)
        n_tiles_y = int((height + tile_size_y - 1) / tile_size_y)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for i in range(n_tiles_x):
            for j in range(n_tiles_y):
                x = i * tile_size_x
                y = j * tile_size_y

                win = Window(
                    col_off=x, row_off=y, width=tile_size_x, height=tile_size_y
                )

                data = src.read(window=win)

                if (data == 0).sum() > (data.size / 2):
                    continue

                new_transform = rasterio.windows.transform(win, transform)

                center_lat, center_lon = get_patch_center_latlon(
                    new_transform, tile_size_x, tile_size_y, src.crs
                )

                tile = get_tile_from_coord(center_lat, center_lon, ZOOM_LEVEL)
                # join_tifs(tile, os.path.join(output_folder, f"tile_{i}_{j}"))

                new_meta = src.meta.copy()
                new_meta.update(
                    {
                        "dtype": src.dtypes[0],
                        "height": tile_size_y,
                        "width": tile_size_x,
                        "transform": new_transform,
                    }
                )

                with rasterio.open(
                    os.path.join(output_folder, f"tile_{i}_{j}.tif"), "w", **new_meta
                ) as dst:
                    dst.write(data)

                # Explicitly delete the data object to release memory.
                del data
                #  force the garbage collector to release unused memory.
                gc.collect()


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

            del memfile

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

    i_path = i_path + f"_sat_{ZOOM_LEVEL}.tif"
    with rasterio.open(i_path, "w", **out_meta) as dest:
        dest.write(mosaic)
        print(f"Saved {i_path} successfully!")

    for t in tile_data:  # Close all datasets
        t.close()

    del neighbors
    del tile_data
    del mosaic
    del out_meta

    gc.collect()


def process_tif(tif):
    input_tiff = "../castral_dataset/RGB/" + tif
    output_directory = f"../castral_dataset/preprocessed/{tif[:-4]}/"
    split_tiff_rasterio(input_tiff, output_directory, 2000, 2000)


if __name__ == "__main__":
    tif_files = [
        tif for tif in os.listdir("../castral_dataset/RGB/") if tif.endswith(".tif")
    ]

    with Pool(4) as pool:
        pool.map(process_tif, tif_files)
