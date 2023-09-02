#! /usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
    satellite_pics = {
        "Zagreb, gozdna površina": "./res/Train3_Zagreb_150m_80fov_90deg_0016_grid.png",
        "Ljubljana, jedro mesta": "./res/Test1_Ljubljana_150m_80fov_90deg_0023_grid.png",
        "Benetke, mesto na morju": "./res/Train10_Venice_150m_80fov_90deg_0018_grid.png",
        "Pordenone, travnata površina": "./res/Train8_Pordenone_150m_80fov_90deg_0138_grid.png",
        "Szombathely, industrija": "./res/Train9_Szombathely_150m_80fov_90deg_0194_grid.png",
        "Pula, mesto ob morju": "./res/Train7_Pula_150m_80fov_90deg_0101_grid.png",
        "Udine, primer pokopališča": "./res/Train6_Udine_150m_80fov_90deg_0197_grid.png",
        "Gradec, kmetijske površine": "./res/Train4_Graz_150m_80fov_90deg_0218_grid.png",
    }

    for grid_idx in range(2):
        fig, axs = plt.subplots(2, 2, figsize=(16, 16))

        for idx, (title, path) in enumerate(
            list(satellite_pics.items())[grid_idx * 4 : (grid_idx + 1) * 4]
        ):
            ax = axs[idx // 2, idx % 2]
            img = mpimg.imread(path)
            ax.imshow(img)
            ax.set_title(title, fontsize=16)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(f"./res/corresponding_sat_examples_grid{grid_idx+1}.png")


if __name__ == "__main__":
    main()
