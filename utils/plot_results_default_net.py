#! /usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
    images_dict = {
        "good1": [
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/sat_overlay_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-4-8.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/drone_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-4-8.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/sat_overlay_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-1-11.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/drone_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-1-11.png",
        ],
        "good2": [
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/sat_overlay_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-2-6.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/drone_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-2-6.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/sat_overlay_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-3-4.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/drone_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-3-4.png",
        ],
        "bad1": [
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/sat_overlay_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-2-9.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/drone_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-2-9.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/sat_overlay_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-2-9.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/drone_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-2-9.png",
        ],
        "bad2": [
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/sat_overlay_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-5-5.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/drone_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-5-5.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/sat_overlay_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-16-4.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/drone_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-16-4.png",
        ],
    }

    def plot_quadruple(drone_path1, sat_path1, drone_path2, sat_path2, count):
        fig, axs = plt.subplots(1, 4, figsize=(30, 8))

        drone_img1 = mpimg.imread(drone_path1)
        axs[0].imshow(drone_img1)
        axs[0].set_title("Slika iz brezpilotnega letalnika", fontsize=16)
        axs[0].axis("off")

        sat_img1 = mpimg.imread(sat_path1)
        axs[1].imshow(sat_img1)
        axs[1].set_title("Izhod modela", fontsize=16)
        axs[1].axis("off")

        drone_img2 = mpimg.imread(drone_path2)
        axs[2].imshow(drone_img2)
        axs[2].set_title("Slika iz brezpilotnega letalnika", fontsize=16)
        axs[2].axis("off")

        sat_img2 = mpimg.imread(sat_path2)
        axs[3].imshow(sat_img2)
        axs[3].set_title("Izhod modela", fontsize=16)
        axs[3].axis("off")

        axs[0].set_position([0.05, 0.125, 0.2, 0.775])  # [left, bottom, width, height]
        axs[1].set_position([0.255, 0.125, 0.2, 0.775])
        axs[2].set_position([0.55, 0.125, 0.2, 0.775])
        axs[3].set_position([0.755, 0.125, 0.2, 0.775])

        plt.savefig(f"./res/drone_net_example_{count}.png", dpi=140)
        plt.close()

    count = 0
    for category, paths in images_dict.items():
        for i in range(0, len(paths), 4):  # We take in groups of 4 now
            plot_quadruple(paths[i + 1], paths[i], paths[i + 3], paths[i + 2], count)
            count += 1


if __name__ == "__main__":
    main()
