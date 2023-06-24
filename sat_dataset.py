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


class SatDataset(Dataset):
    def __init__(self, root_dir="./sat/", patch_w=512, patch_h=512, zoom_level=16):
        self.root_dir = root_dir
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.zoom_level = zoom_level
        self.metadata_dict = {}
        self.image_paths = self.get_entry_paths(self.root_dir)
        self.bboxes = {
            "Ljubljana": [14.4, 46.0, 14.6, 46.1],  # min_lon, min_lat, max_lon, max_lat
            "Maribor": [15.6, 46.5, 15.7, 46.6],
            "Koper": [13.5484, 45.5452, 13.6584, 45.6052],
            "Trieste": [13.7386, 45.6333, 13.8486, 45.7033],
            "Graz": [15.4294, 47.0595, 15.5394, 47.1195],
            "Pordenone": [12.6496, 45.9564, 12.7596, 46.0164],
            "Udine": [13.2180, 46.0614, 13.3280, 46.1214],
            "Klagenfurt": [14.2765, 46.6203, 14.3865, 46.6803],
            "Pula": [13.8490, 44.8683, 13.9590, 44.9283],
            "Szombathely": [16.6056, 47.2256, 16.7156, 47.2856],
            "Venice": [12.3155, 45.4408, 12.4255, 45.5008],
        }

        self.download_maps()
        self.fill_metadata_dict()

    def fill_metadata_dict(self):
        for image_path in self.image_paths:
            img_info = self.extract_info_from_filename(image_path)
            self.metadata_dict[img_info] = image_path

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
        image = Image.open(image_path)

        img_info = self.extract_info_from_filename(image_path)
        image = np.array(image, dtype=np.float32)
        image = image / 255.0
        image = torch.from_numpy(image)

        return image, img_info

    def extract_info_from_filename(self, filename):
        # 16_35582_23023.jpg -> z, x, y
        fn = filename.split("/")[-1]
        fn = fn.split(".")[0]
        fn = fn.split("_")
        try:
            z, x, y = int(fn[0]), int(fn[1]), int(fn[2])
        except:
            print("Error extracting info from the filename in sat images: ", filename)
            return None

        return mercantile.Tile(x, y, z)

        ### -------------------------- ###
        ### MAP MANIPULATION FUNCTIONS ###
        ### -------------------------- ###

    def find_tile(self, lat, lng):
        for tile in self.metadata_dict.keys():
            if self.is_coord_in_a_tile(lat, lng, tile):
                fpath = self.metadata_dict[tile]
                ix = self.image_paths.index(fpath)
                return self.__getitem__(ix)

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

        for r_name, bbox in self.bboxes.items():
            print("Downloading maps for region: ", r_name)
            for tile in tqdm(bbox[0], bbox[1], bbox[2], bbox[3], self.zoom_level):
                box = mercantile.bounds(tile)

                center_lat = (box.north + box.south) / 2
                center_lng = (box.east + box.west) / 2

                os.makedirs(f"{self.root_dir}/tiles/{r_name}", exist_ok=True)
                file_path = f"{self.root_dir}/tiles/{r_name}/{self.zoom_level}_{tile.x}_{tile.y}.jpg"

                if not os.path.exists(file_path):
                    response = requests.get(
                        f"https://c.tiles.mapbox.com/v4/mapbox.satellite/{self.zoom_level}/{tile.x}/{tile.y}@2x.jpg",
                        params=params,
                        headers=headers,
                    )

                    if response.status_code == 200:
                        with open(file_path, "wb") as f:
                            f.write(response.content)
                    else:
                        print(f"Error downloading {file_path}")


def main():
    pass


if __name__ == "__main__":
    main()
