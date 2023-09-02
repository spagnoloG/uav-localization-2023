#! /usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
    pics = {
        "Zagreb, gozdna površina": "../dataset/Train3_Zagreb_150m_80fov_90deg/footage/Train3_Zagreb_150m_80fov_90deg_0016.jpeg",
        "Ljubljana, jedro mesta": "../dataset/Test1_Ljubljana_150m_80fov_90deg/footage/Test1_Ljubljana_150m_80fov_90deg_0205.jpeg",
        "Benetke, mesto na morju": "../dataset/Train10_Venice_150m_80fov_90deg/footage/Train10_Venice_150m_80fov_90deg_0018.jpeg",
        "Pordenone, travnata površina": "../dataset/Train8_Pordenone_150m_80fov_90deg/footage/Train8_Pordenone_150m_80fov_90deg_0138.jpeg",
        "Szombathely, industrija": "../dataset/Train9_Szombathely_150m_80fov_90deg/footage/Train9_Szombathely_150m_80fov_90deg_0194.jpeg",
        "Pula, mesto ob morju": "../dataset/Train7_Pula_150m_80fov_90deg/footage/Train7_Pula_150m_80fov_90deg_0101.jpeg",
        "Udine, primer pokopališča": "../dataset/Train6_Udine_150m_80fov_90deg/footage/Train6_Udine_150m_80fov_90deg_0197.jpeg",
        "Gradec, kmetijske površine": "../dataset/Train4_Graz_150m_80fov_90deg/footage/Train4_Graz_150m_80fov_90deg_0218.jpeg",
    }

    for grid_idx in range(2):
        fig, axs = plt.subplots(2, 2, figsize=(16, 9))

        for idx, (title, path) in enumerate(
            list(pics.items())[grid_idx * 4 : (grid_idx + 1) * 4]
        ):
            ax = axs[idx // 2, idx % 2]
            img = mpimg.imread(path)
            ax.imshow(img)
            ax.set_title(title, fontsize=16)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(f"./res/drone_examples_grid{grid_idx+1}.png", dpi=80)


if __name__ == "__main__":
    main()
