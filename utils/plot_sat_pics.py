#! /usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
    satellite_pics = {
        "Zagreb, gozdna površina": "./res/Train3_Zagreb_150m_80fov_90deg_0016_grid.png",
        "Ljubljana, jedro mesta": "./res/Test1_Ljubljana_150m_80fov_90deg_0023_grid.png",
        "Benetke, mesto na morju": "./res/Train10_Venice_150m_80fov_90deg_0018_grid.png",
        "Pordenone, travnata površina": "./res/Train8_Pordenone_150m_80fov_90deg_0138_grid.png",
    }

    fig, axs = plt.subplots(2, 2, figsize=(16, 16))

    for idx, (title, path) in enumerate(satellite_pics.items()):
        ax = axs[idx // 2, idx % 2]
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(title, fontsize=16)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("./res/corresponding_sat_examples.png")


if __name__ == "__main__":
    main()
