#!/usr/bin/env python3

import mercantile
import requests
import os
import csv

from bounding_boxes import bboxes

zoom_level = 16
metadata_file = "metadata.csv"
csv_file = open(metadata_file, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(
    [
        "file_path",
        "lat",
        "lng",
        "bbox_west",
        "bbox_east",
        "bbox_north",
        "bbox_south",
        "x",
        "y",
        "z",
    ]
)


# Create a directory to store the downloaded tiles
os.makedirs("tiles", exist_ok=True)

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


for r_name, bbox in bboxes.items():
    for tile in mercantile.tiles(bbox[0], bbox[1], bbox[2], bbox[3], zoom_level):
        bbox = mercantile.bounds(tile)

        center_lat = (bbox.north + bbox.south) / 2
        center_lng = (bbox.east + bbox.west) / 2

        os.makedirs(f"tiles/{r_name}", exist_ok=True)
        file_path = f"tiles/{r_name}/{zoom_level}_{tile.x}_{tile.y}.jpg"

        csv_writer.writerow(
            [
                file_path,
                center_lat,
                center_lng,
                bbox.west,
                bbox.east,
                bbox.north,
                bbox.south,
                tile.x,
                tile.y,
                zoom_level,
            ]
        )

        if not os.path.exists(file_path):
            response = requests.get(
                f"https://c.tiles.mapbox.com/v4/mapbox.satellite/{zoom_level}/{tile.x}/{tile.y}@2x.jpg",
                params=params,
                headers=headers,
            )

            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(response.content)
            else:
                print(f"Error downloading {file_path}")

csv_file.close()
