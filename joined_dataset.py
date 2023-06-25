from drone_dataset import DroneDataset
from sat_dataset import SatDataset
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class JoinedDataset(Dataset):
    def __init__(self, drone_dir="./drone/", sat_dir="./sat/"):
        self.drone_dataset = DroneDataset(root_dir=drone_dir)
        self.sat_dataset = SatDataset(root_dir=sat_dir)

    def __len__(self):
        return len(self.drone_dataset)

    def __getitem__(self, idx):
        drone_image, drone_info = self.drone_dataset[idx]

        lat = drone_info["coordinate"]["latitude"]
        lon = drone_info["coordinate"]["longitude"]

        print(lat, lon)

        # Get corresponding satellite image
        sat_image, sat_info = self.sat_dataset.find_tile(lat, lon)
        print(sat_info)

        return drone_image, drone_info, sat_image, sat_info


def test():
    dataloader = DataLoader(
        JoinedDataset(drone_dir="./drone/", sat_dir="./sat/"),
        batch_size=10,
        shuffle=True,
    )

    # Fetch one batch of data
    drone_images, drone_infos, sat_images, sat_infos = next(iter(dataloader))

    # Plot first 5 pairs of images
    for i in range(5):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Plot drone image
        drone_image = drone_images[i].numpy()
        ax1.imshow(drone_image)
        ax1.set_title(f"Drone Image {i+1}")
        ax1.axis("off")

        # Plot satellite image
        sat_image = sat_images[i].numpy()
        ax2.imshow(sat_image)
        ax2.set_title(f"Satellite Image {i+1}")
        ax2.axis("off")

        plt.show()


if __name__ == "__main__":
    test()
