#! /usr/bin/env python3

import cv2
import os

DRONE_DATASET_DIR = "../dataset/"


def is_green_region(image_path, threshold=0.51):
    image = cv2.imread(image_path)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for green color
    lower_green = (30, 40, 40)
    upper_green = (90, 255, 255)

    # Create a mask for green regions
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Calculate the percentage of green pixels
    total_pixels = image.size / 3
    green_pixels = cv2.countNonZero(green_mask)

    green_ratio = green_pixels / total_pixels

    return green_ratio > threshold


def extract_folder_name(image_path):
    return image_path.split("/")[-3]


def get_image_paths():
    for root, dirs, files in os.walk(DRONE_DATASET_DIR):
        for file in files:
            if file.endswith(".jpeg"):
                image_path = os.path.join(root, file)
                yield image_path, extract_folder_name(image_path)


def classify_images(res_dict):
    for image_path, region in get_image_paths():
        if region not in res_dict:
            res_dict[region] = {"green-field": 0, "building": 0}
        if is_green_region(image_path):
            res_dict[region]["green-field"] += 1
        else:
            res_dict[region]["building"] += 1


def main():
    res_dict = {}
    classify_images(res_dict)
    with open("./res/region_structures.json", "w") as f:
        json.dump(res_dict, f, indent=4)

if __name__ == "__main__":
    main()
