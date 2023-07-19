import os
import csv


def get_entry_paths(path):
    entry_paths = []
    entries = os.listdir(path)
    for entry in entries:
        entry_path = path + "/" + entry
        if os.path.isdir(entry_path):
            entry_paths += get_entry_paths(entry_path)
        if entry_path.endswith(".jpg"):
            entry_paths.append(entry_path)
    return entry_paths


def main():
    with open("metadata.csv", "w", newline="") as csvfile:
        # Write header
        fieldnames = ["file_path", "x", "y", "z"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for entry_path in get_entry_paths("./tiles"):
            # Get x, y, z
            ep = entry_path.replace("./tiles/", "")
            ep = entry_path.replace(".jpg", "")
            z, x, y = ep.split("/")[3].split("_")
            z = int(z)
            x = int(x)
            y = int(y)
            # Write row
            writer.writerow({"file_path": entry_path, "x": x, "y": y, "z": z})


if __name__ == "__main__":
    main()
