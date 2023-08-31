#! /usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
    pics = {
        "Zagreb, gozdna površina": "../dataset/Train3_Zagreb_150m_80fov_90deg/footage/Train3_Zagreb_150m_80fov_90deg_0016.jpeg",
        "Ljubljana, jedro mesta": "../dataset/Test1_Ljubljana_150m_80fov_90deg/footage/Test1_Ljubljana_150m_80fov_90deg_0205.jpeg",
        "Benetke, mesto ob morju": "../dataset/Train10_Venice_150m_80fov_90deg/footage/Train10_Venice_150m_80fov_90deg_0018.jpeg",
        "Pordenone, travnata površina": "../dataset/Train8_Pordenone_150m_80fov_90deg/footage/Train8_Pordenone_150m_80fov_90deg_0138.jpeg",
    }

    fig, axs = plt.subplots(2, 2, figsize=(16, 9))

    for idx, (title, path) in enumerate(pics.items()):
        ax = axs[idx // 2, idx % 2]
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(title, fontsize=16)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("./res/drone_examples.png", dpi=70)


if __name__ == "__main__":
    main()
