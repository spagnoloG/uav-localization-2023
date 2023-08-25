import csv
import numpy as np
import matplotlib.pyplot as plt


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

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot RDS data on primary y-axis
    ax1.plot(hann_kernels, rds_train, "-o", label="RDS Train", color="b")
    ax1.plot(hann_kernels, rds_val, "-s", label="RDS Val", color="c")
    ax1.plot(
        hann_kernels[rds_train_best_idx],
        rds_train[rds_train_best_idx],
        "b*",
        markersize=12,
    )
    ax1.plot(
        hann_kernels[rds_val_best_idx], rds_val[rds_val_best_idx], "c*", markersize=12
    )
    ax1.set_xlabel("Velikost jedra", fontsize=14)
    ax1.set_ylabel("RDS", fontsize=14, color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # Create secondary y-axis for Hann values
    ax2 = ax1.twinx()
    ax2.plot(hann_kernels, hann_train, "-o", label="Hann Train", color="r")
    ax2.plot(hann_kernels, hann_val, "-s", label="Hann Val", color="m")
    ax2.plot(
        hann_kernels[hann_train_best_idx],
        hann_train[hann_train_best_idx],
        "r*",
        markersize=12,
    )
    ax2.plot(
        hann_kernels[hann_val_best_idx],
        hann_val[hann_val_best_idx],
        "m*",
        markersize=12,
    )
    ax2.set_ylabel("Hanningova izguba", fontsize=14, color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    ax2.legend(loc="upper right")

    fig.suptitle(
        "Primerjava rezultatov ob uporabi razlicne velikosti Hanningovega okna",
        fontsize=12,
    )

    # Save the figure
    plt.savefig("./res/plot_different_hann_kernels.png", dpi=200)


if __name__ == "__main__":
    main()
