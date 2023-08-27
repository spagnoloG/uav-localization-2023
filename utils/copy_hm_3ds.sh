#! /usr/bin/env bash

set -eux -o pipefail

CSV_FILE="./res/hann_kers.csv"
TARGET_DIR="./res/heatmaps3d"

if [ ! -f "$CSV_FILE" ]; then
    echo "$CSV_FILE not found!"
    exit 1
fi

mkdir -p "$TARGET_DIR"

# Read the CSV line by line, skipping the header
tail -n +2 "$CSV_FILE" | while IFS=", " read -r size hash; do
    SOURCE_FILE="../vis/$hash/validation_3d_hm_${hash}-3-1.png"

    if [ -f "$SOURCE_FILE" ]; then
        cp "$SOURCE_FILE" "$TARGET_DIR/"
        echo "Copied $SOURCE_FILE to $TARGET_DIR/"
    else
        echo "File $SOURCE_FILE not found!"
    fi
done
