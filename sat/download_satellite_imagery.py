#!/usr/bin/env python3

import mercantile
import requests
import os
from tqdm import tqdm
import time

from bounding_boxes import bboxes

zoom_level = 16

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


def downloader():
    os.makedirs(f"./tiles", exist_ok=True)
    for r_name, bbox in bboxes.items():
        print("Downloading tiles for", r_name)
        for tile in tqdm(
            mercantile.tiles(bbox[0], bbox[1], bbox[2], bbox[3], zoom_level)
        ):
            box = mercantile.bounds(tile)

            center_lat = (box.north + box.south) / 2
            center_lng = (box.east + box.west) / 2

            file_path = "./tiles/" + f"{zoom_level}_{tile.x}_{tile.y}.jpg"

            if not os.path.exists(file_path):
                max_attempts = 5
                for attempt in range(max_attempts):
                    try:
                        response = requests.get(
                            f"https://c.tiles.mapbox.com/v4/mapbox.satellite/{zoom_level}/{tile.x}/{tile.y}@2x.jpg",
                            params=params,
                            headers=headers,
                        )
                        response.raise_for_status()  # raises a Python exception if the response contains an HTTP error status code
                    except (
                        requests.exceptions.RequestException,
                        requests.exceptions.ConnectionError,
                    ) as e:
                        if (
                            attempt < max_attempts - 1
                        ):  # i.e., if it's not the final attempt
                            print(
                                f"Attempt {attempt + 1} of {max_attempts} failed. Retrying..."
                            )
                            time.sleep(5)  # wait for 5 seconds before trying again
                            continue
                        else:
                            print(f"Error downloading {file_path}: {e}")
                            break
                    else:  # executes if the try block didn't throw any exceptions
                        with open(file_path, "wb") as f:
                            f.write(response.content)
                        break
                else:
                    print(f"Error downloading {file_path}: {e}")


if __name__ == "__main__":
    downloader()
