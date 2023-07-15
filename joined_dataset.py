from drone_dataset import DroneDataset
from sat_dataset import SatDataset
from sat_dataset import MapUtils
from torch.utils.data import Dataset, DataLoader
import torch


class JoinedDataset(Dataset):
    """
    Dataset that combines drone and satellite images.

    Args:
        drone_dir (str): Directory path for the drone images.
        sat_dir (str): Directory path for the satellite images.
        dataset (str): Dataset type ("train" or "test").
        download_dataset (bool): Flag indicating whether to download the dataset.
        config (dict): Configuration parameters for the drone and satellite datasets.
        heatmap_kernel_size (int): Size of the heatmap kernel.
        drone_view_patch_size (int): Patch size for the drone view.
        metadata_rtree_index (rtree.index.Index): R-tree index for satellite metadata.

    """

    def __init__(
        self,
        drone_dir="./drone/",
        sat_dir="./sat/",
        dataset="train",
        download_dataset=False,
        config=None,
        heatmap_kernel_size=110,
        drone_view_patch_size=128,
        metadata_rtree_index=None,
    ):
        self.heatmap_kernel_size = heatmap_kernel_size
        self.download_dataset = download_dataset
        self.metadata_rtree_index = metadata_rtree_index
        self.drone_dataset = DroneDataset(
            dataset=dataset,
            config=config["drone_dataset"],
            patch_h=drone_view_patch_size,
            patch_w=drone_view_patch_size,
        )
        self.sat_dataset = SatDataset(
            config=config["sat_dataset"],
            download_dataset=download_dataset,
            heatmap_kernel_size=heatmap_kernel_size,
            metadata_rtree_index=self.metadata_rtree_index,
        )
        self.drone_resolution = (self.drone_dataset.patch_w, self.drone_dataset.patch_h)
        self.satellite_resolution = (self.sat_dataset.patch_w, self.sat_dataset.patch_h)
        self.metadata_rtree_index = self.sat_dataset.metadata_rtree_index

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.

        """
        return len(self.drone_dataset)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset at the given index.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: Tuple containing the drone image, drone info, satellite image, satellite info, and heatmap.

        """
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
    import yaml
    from matplotlib import pyplot as plt

    with open("./conf/configuration.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataloader = DataLoader(
        JoinedDataset(config=config),
        batch_size=10,
        shuffle=True,
    )

    # Fetch one batch of data
    drone_images, drone_infos, sat_images, sat_infos, heatmap = next(iter(dataloader))

    ## Plot first 5 pairs of images
    for i in range(5):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Plot drone image
        drone_image = drone_images[i].permute(1, 2, 0).numpy()
        ax1.imshow(drone_image)
        ax1.set_title(f"Drone Image {i+1}")
        ax1.axis("off")

        # Plot satellite image
        sat_image = sat_images[i].permute(1, 2, 0).numpy()
        ax2.imshow(sat_image)
        ax2.set_title(f"Satellite Image {i+1}")
        ax2.axis("off")

        plt.show()


if __name__ == "__main__":
    test()
