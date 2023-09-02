import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    with open("./res/results.csv", "r") as f:
        reader = csv.reader(f)
        data = list(reader)

    data = data[1:]

    hann_kernels = np.array([float(row[0]) for row in data])
    rds_train = np.array([float(row[1]) for row in data])
    rds_val = np.array([float(row[2]) for row in data])
    hann_train = np.array([float(row[3]) for row in data])
    hann_val = np.array([float(row[4]) for row in data])

    # Find the best scores
    rds_train_best_idx = np.argmax(rds_train)
    rds_val_best_idx = np.argmax(rds_val)
    hann_train_best_idx = np.argmin(hann_train)
    hann_val_best_idx = np.argmin(hann_val)

    # Set Seaborn style
    sns.set_style("whitegrid")

    # Use Seaborn's color palette
    colors = sns.color_palette("deep")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot RDS data on primary y-axis using Seaborn colors
    ax1.plot(hann_kernels, rds_train, "-o", label="RDS Train", color=colors[0])
    ax1.plot(hann_kernels, rds_val, "-s", label="RDS Val", color=colors[1])
    ax1.plot(
        hann_kernels[rds_train_best_idx],
        rds_train[rds_train_best_idx],
        "*",
        color=colors[0],
        markersize=12,
    )
    ax1.plot(
        hann_kernels[rds_val_best_idx],
        rds_val[rds_val_best_idx],
        "*",
        color=colors[1],
        markersize=12,
    )
    ax1.set_xlabel("Velikost jedra", fontsize=14)
    ax1.set_ylabel("RDS", fontsize=14, color=colors[0])
    ax1.tick_params(axis="y", labelcolor=colors[0])
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # Adjust x-axis to show odd numbers
    ax1.set_xticks(
        [
            x
            for x in range(int(min(hann_kernels)), int(max(hann_kernels)) + 1)
            if x % 2 != 0
        ]
    )

    # Create secondary y-axis for Hann values using Seaborn colors
    ax2 = ax1.twinx()
    ax2.plot(hann_kernels, hann_train, "-o", label="Hann Train", color=colors[2])
    ax2.plot(hann_kernels, hann_val, "-s", label="Hann Val", color=colors[3])
    ax2.plot(
        hann_kernels[hann_train_best_idx],
        hann_train[hann_train_best_idx],
        "*",
        color=colors[2],
        markersize=12,
    )
    ax2.plot(
        hann_kernels[hann_val_best_idx],
        hann_val[hann_val_best_idx],
        "*",
        color=colors[3],
        markersize=12,
    )
    ax2.set_ylabel("Hanningova izguba", fontsize=14, color=colors[2])
    ax2.tick_params(axis="y", labelcolor=colors[2])
    ax2.legend(loc="upper right")

    fig.suptitle(
        "Primerjava rezultatov ob uporabi razlicne velikosti Hanningovega okna",
        fontsize=12,
    )

    plt.savefig("./res/plot_different_hann_kernels.png", dpi=220)


if __name__ == "__main__":
    main()
