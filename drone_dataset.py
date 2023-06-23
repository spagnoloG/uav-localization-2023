import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import numpy.typing as npt
from itertools import chain
import matplotlib.pyplot as plt


class DroneImagePreprocess:
    def __init__(
        self,
        path="./drone/",
        validation_set="Train1_MB_150m_80fov_90deg",
        test_set="Train1_MB_150m_80fov_90deg",
        patch_w=750,
        patch_h=750,
    ):
        super().__init__()
        self.path = path
        self.training_set = []
        self.validation_set = []
        self.test_set = []
        self.entry_paths = []
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.metadata_dict = {}

    def load_and_preprocess_images(self):
        self.get_entry_paths(self.path)
        for entry_path in self.entry_paths:
            result = self.load_image_helper(entry_path)
            if result is not None:
                image, img_info = result
                preprocessed_images, preprocessed_infos = self.preprocess_image(image, img_info)
                for img, info in zip(preprocessed_images,
                                     preprocessed_infos):
                    yield img, info

    #def load_images(self):
    #    self.get_entry_paths(self.path)
    #    load_image_partial = partial(self.load_image_helper)
    #    preprocess_image_partial = partial(self.preprocess_image)

    #    with Pool() as pool:
    #        results = pool.map(load_image_partial, self.entry_paths)

    #    X = [result[0] for result in results if result is not None]
    #    y = [result[1] for result in results if result is not None]

    #    with Pool() as pool:
    #        processed_results = pool.map(preprocess_image_partial, zip(X, y))

    #    # Flatten the lists
    #    X_processed = list(chain.from_iterable([result[0] for result in processed_results]))
    #    y_processed = list(chain.from_iterable([result[1] for result in processed_results]))

    #    return X_processed, y_processed


    def load_image_helper(self, entry_path) -> tuple[npt.NDArray, str]:
        try:
            img = Image.open(entry_path)
        except PIL.UnidentifiedImageError as e:
            print("Could not open an image: ", entry_path)
            print(e)
            return None

        lookup_str, file_number = self.extract_info_from_filename(entry_path)

        if lookup_str is None:
            return None

        img_info = self.metadata_dict[lookup_str][file_number]
        img_info["filename"] = entry_path

        return img, img_info

    def extract_info_from_filename(self, filename) -> tuple[str, int]:
        filename_without_ext = filename.replace(".jpeg", "")
        segments = filename_without_ext.split("/")
        info = segments[-1]
        try:
            number = int(info.split("_")[-1])
        except ValueError:
            print("Could not extract number from filename: ", filename)
            return None

        info = "_".join(info.split("_")[:-1])

        return info, number

    def get_metadata(self, path) -> None:
        with open(path, newline="") as jsonfile:
            json_dict = json.load(jsonfile)
            path = path.split("/")[-1]
            path = path.replace(".json", "")
            self.metadata_dict[path] = json_dict["cameraFrames"]

    def get_entry_paths(self, path) -> list[str]:
        entries = os.listdir(path)
        for entry in entries:
            entry_path = path + "/" + entry
            if os.path.isdir(entry_path):
                self.get_entry_paths(entry_path + "/")
            if entry_path.endswith(".jpeg"):
                self.entry_paths.append(entry_path)
            if entry_path.endswith(".json"):
                self.get_metadata(entry_path)

    def preprocess_image(self, image, img_info) -> tuple[list[npt.NDArray], list[str]]:
        imgs = []
        infos = []
        for angle in range(0, 360, 20):
            img = image.copy().rotate(angle)
            img = self.crop_center(img, self.patch_w, self.patch_h)
            img = np.array(img, dtype=np.float32)
            img = img / 255
            info = img_info.copy()
            info["angle"] = angle
            imgs.append(img)
            infos.append(info)
        return imgs, infos

    def crop_center(self, img, crop_width, crop_height) -> Image:
        img_width, img_height = img.size
        return img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


class GEDataset(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]

        if self.transforms != None:
            image = self.transforms(image)
        return image


def main():
    image_preprocessor = DroneImagePreprocess().load_and_preprocess_images()
    for img, info in image_preprocessor:
        # plot the image for sanity check
        plt.imshow(img)
        plt.show()
        print(info)



if __name__ == "__main__":
    main()
