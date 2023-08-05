import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def hanning_2d(M):
    hanning_window = np.hanning(M)
    return np.sqrt(np.outer(hanning_window, hanning_window))


def gaussian_2d(size, sigma=1):
    ax = np.arange(-size // 2 + 1.0, size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)


def plot_kernel(kernel, title):
    plt.imshow(kernel, cmap="hot", interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.savefig(title + ".jpg", format="jpg")
    plt.show()


# Define size and sigma
size = 33
sigma = 10

# Generate and plot 2D Hanning kernel
hanning_kernel = hanning_2d(size)
plot_kernel(hanning_kernel, "2D Hanning kernel")

# Generate and plot 2D Gaussian kernel
gaussian_kernel = gaussian_2d(size, sigma)
plot_kernel(gaussian_kernel, "2D Gaussian kernel")
