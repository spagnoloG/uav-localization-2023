import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec


def main():
    images_dict = {
        "good1": [
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/sat_overlay_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-0-5.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/drone_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-0-5.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/satellite_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-0-5.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/sat_overlay_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-1-14.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/drone_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-1-14.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/satellite_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-1-14.png",
        ],
        "good2": [
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/sat_overlay_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-5-0.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/drone_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-5-0.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/satellite_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-5-0.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/sat_overlay_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-5-1.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/drone_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-5-1.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/satellite_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-5-1.png",
        ],
        "good3": [
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/sat_overlay_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-4-8.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/drone_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-4-8.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/satellite_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-4-8.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/sat_overlay_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-0-14.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/drone_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-0-14.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/satellite_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-0-14.png",
        ],
        "bad1": [
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/sat_overlay_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-12-1.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/drone_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-12-1.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/satellite_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-12-1.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/sat_overlay_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-17-5.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/drone_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-17-5.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/satellite_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-17-5.png",
        ],
        "bad2": [
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/sat_overlay_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-6-9.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/drone_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-6-9.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/satellite_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-6-9.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/sat_overlay_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-18-11.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/drone_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-18-11.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/satellite_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-18-11.png",
        ],
        "bad3": [
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/sat_overlay_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-24-9.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/drone_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-24-9.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/satellite_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-24-9.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/sat_overlay_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-1-13.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/drone_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-1-13.png",
            "../vis/b5ce014c96d04f16e1ffd457ca46200ed4a4bc03/satellite_image_b5ce014c96d04f16e1ffd457ca46200ed4a4bc03-1-13.png",
        ],
    }

    def plot_quadruple(
        drone_path1,
        sat_path1,
        overlay_path1,
        drone_path2,
        sat_path2,
        overlay_path2,
        count,
    ):
        # Create the figure
        fig = plt.figure(figsize=(30, 8))

        gs = gridspec.GridSpec(1, 7, width_ratios=[1, 2, 2, 0.5, 1, 2, 2])

        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(mpimg.imread(drone_path1))
        ax1.set_title("Drone Image", fontsize=16)
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[1])
        ax2.imshow(mpimg.imread(sat_path1))
        ax2.set_title("Satellite Image", fontsize=16)
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[2])
        ax3.imshow(mpimg.imread(overlay_path1))
        ax3.set_title("Satellite Overlay", fontsize=16)
        ax3.axis("off")

        ax4 = fig.add_subplot(gs[4])
        ax4.imshow(mpimg.imread(drone_path2))
        ax4.set_title("Drone Image", fontsize=16)
        ax4.axis("off")

        ax5 = fig.add_subplot(gs[5])
        ax5.imshow(mpimg.imread(sat_path2))
        ax5.set_title("Satellite Image", fontsize=16)
        ax5.axis("off")

        ax6 = fig.add_subplot(gs[6])
        ax6.imshow(mpimg.imread(overlay_path2))
        ax6.set_title("Satellite Overlay", fontsize=16)
        ax6.axis("off")

        plt.tight_layout()

        plt.savefig(
            f"./res/drone_net_example_{count}.png",
            dpi=100,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close()

    count = 0
    for category, paths in images_dict.items():
        for i in range(0, len(paths), 6):  # Groups of 6
            plot_quadruple(
                paths[i + 1],
                paths[i + 2],
                paths[i],
                paths[i + 4],
                paths[i + 5],
                paths[i + 3],
                count,
            )
            count += 1


if __name__ == "__main__":
    main()
