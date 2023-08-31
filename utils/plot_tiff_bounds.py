import matplotlib.pyplot as plt


def plot_with_grid(image, fname="grid.png"):
    # Create a new figure with the exact dimensions of the image
    fig = plt.figure(figsize=(image.shape[1] / 80.0, image.shape[0] / 80.0), dpi=60)

    # Add an axis that spans the entire figure
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    ax.imshow(image, origin="upper")

    # Calculate grid spacing based on image dimensions
    x_spacing = image.shape[1] // 3
    y_spacing = image.shape[0] // 3

    grid_color = "#d3d3d3"  # Light gray
    grid_alpha = 0.5  # Semi-transparent
    grid_linestyle = "--"  # Dashed lines
    grid_linewidth = 5

    # Draw vertical grid lines
    for x in range(x_spacing, image.shape[1], x_spacing):
        ax.vlines(
            x,
            0,
            image.shape[0],
            colors=grid_color,
            alpha=grid_alpha,
            linestyle=grid_linestyle,
            linewidth=grid_linewidth,
        )

    # Draw horizontal grid lines
    for y in range(y_spacing, image.shape[0], y_spacing):
        ax.hlines(
            y,
            0,
            image.shape[1],
            colors=grid_color,
            alpha=grid_alpha,
            linestyle=grid_linestyle,
            linewidth=grid_linewidth,
        )

    # Save the figure without borders
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(f"./res/{fname}", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.show()
    plt.cla()
    plt.clf()


def main():
    tiff_paths = [
        "../dataset/Train3_Zagreb_150m_80fov_90deg/footage/Train3_Zagreb_150m_80fov_90deg_0016.jpeg_sat_16.tiff",
        "../dataset/Test1_Ljubljana_150m_80fov_90deg/footage/Test1_Ljubljana_150m_80fov_90deg_0205.jpeg_sat_16.tiff",
        "../dataset/Train10_Venice_150m_80fov_90deg/footage/Train10_Venice_150m_80fov_90deg_0018.jpeg_sat_16.tiff",
        "../dataset/Train8_Pordenone_150m_80fov_90deg/footage/Train8_Pordenone_150m_80fov_90deg_0138.jpeg_sat_16.tiff",
    ]

    for p in tiff_paths:
        img = plt.imread(p)
        fname = p.split("/")[-1].split(".")[0] + "_grid.png"
        plot_with_grid(img, fname)


if __name__ == "__main__":
    main()
