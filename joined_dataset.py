from drone_dataset import DroneDataset
from sat_dataset import SatDataset
from sat_dataset import MapUtils
from torch.utils.data import Dataset, DataLoader
import torch


class JoinedDataset(Dataset):
    def __init__(
        self,
        drone_dir="./drone/",
        sat_dir="./sat/",
        dataset="train",
        download_dataset=False,
        config=None,
        heatmap_kernel_size=110,
        drone_view_patch_size=128,
    ):
        self.heatmap_kernel_size = heatmap_kernel_size
        self.download_dataset = download_dataset
        self.drone_dataset = DroneDataset(
            dataset=dataset,
            config=config["drone_dataset"],
            patch_h=drone_view_patch_size,
            patch_w=drone_view_patch_size,
        )
        self.sat_dataset = SatDataset(
            config=config["sat_dataset"], download_dataset=download_dataset
        )
        self.drone_resolution = (self.drone_dataset.patch_w, self.drone_dataset.patch_h)
        self.satellite_resolution = (self.sat_dataset.patch_w, self.sat_dataset.patch_h)

    def __len__(self):
        return len(self.drone_dataset)

    def __getitem__(self, idx):
        drone_image, drone_info = self.drone_dataset[idx]

        lat = drone_info["coordinate"]["latitude"]
        lon = drone_info["coordinate"]["longitude"]

        # Get corresponding satellite image
        sat_image, sat_info = self.sat_dataset.find_tile(lat, lon)

        heatmap = self.sat_dataset.generate_heatmap(
            lat,
            lon,
            sat_image,
            sat_info[0],
            sat_info[1],
            sat_info[2],
            self.heatmap_kernel_size,
        )

        return drone_image, drone_info, sat_image, sat_info, heatmap


def test():
    dataloader = DataLoader(
        JoinedDataset(drone_dir="./drone/", sat_dir="./sat/"),
        batch_size=10,
        shuffle=True,
    )

    # Fetch one batch of data
    drone_images, drone_infos, sat_images, sat_infos = next(iter(dataloader))

    ## Plot first 5 pairs of images
    # for i in range(5):
    # fig, (ax1, ax2) = plt.subplots(1, 3, figsize=(10, 5))

    # Plot drone image
    # drone_image = drone_images[i].numpy()
    # ax1.imshow(drone_image)
    # ax1.set_title(f"Drone Image {i+1}")
    # ax1.axis("off")

    # Plot satellite image
    # sat_image = sat_images[i].numpy()
    # ax2.imshow(sat_image)
    # ax2.set_title(f"Satellite Image {i+1}")
    # ax2.axis("off")

    # plt.show()


if __name__ == "__main__":
    test()
