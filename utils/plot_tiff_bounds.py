import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

img_path = "../dataset/Test1_Ljubljana_150m_80fov_90deg/footage/Test1_Ljubljana_150m_80fov_90deg_0023.jpeg"
tiff_path = "../dataset/Test1_Ljubljana_150m_80fov_90deg/footage/Test1_Ljubljana_150m_80fov_90deg_0023.jpeg_sat_16.tiff"


def plot_with_grid(image):
    # Create a new figure with the exact dimensions of the image
    fig = plt.figure(figsize=(image.shape[1] / 80.0, image.shape[0] / 80.0), dpi=200)

    # Add an axis that spans the entire figure
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)

    # Hide the axis
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    # Display the image
    ax.imshow(image, origin="upper")

    # Calculate grid spacing based on image dimensions
    x_spacing = image.shape[1] // 3
    y_spacing = image.shape[0] // 3

    # Draw vertical grid lines
    for x in range(x_spacing, image.shape[1], x_spacing):
        ax.vlines(x, 0, image.shape[0], colors="r", linewidth=2)

    # Draw horizontal grid lines
    for y in range(y_spacing, image.shape[0], y_spacing):
        ax.hlines(y, 0, image.shape[1], colors="r", linewidth=2)

    # Save the figure without borders
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig("grid.png", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.show()


image = plt.imread(tiff_path)

plot_with_grid(image)
