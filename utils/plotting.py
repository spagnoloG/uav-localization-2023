import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

def hanning_2d(M):
    """
    Return normalized hanning window
    """
    hanning_window = np.hanning(M)
    hw = np.sqrt(np.outer(hanning_window, hanning_window))
    hw = hw / np.sum(hw)
    return hw


def gaussian_2d(size, sigma=1):
    """
    Return normalized 2D Gaussian kernel
    """
    ax = np.arange(-size // 2 + 1.0, size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)

def plot_kernel_3d(kernel, title):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.linspace(0, kernel.shape[1]-1, kernel.shape[1])
    y = np.linspace(0, kernel.shape[0]-1, kernel.shape[0])
    X, Y = np.meshgrid(x, y)
    
    surf = ax.plot_surface(X, Y, kernel, cmap='jet', linewidth=0, antialiased=True)
    
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)
    
    ax.set_title(title)
    plt.savefig("./res/" + title + ".png", format="png")
    plt.show()
    plt.close()

# Define size and sigma
size = 33
sigma = 10

# Generate and plot 2D Hanning kernel
hanning_kernel = hanning_2d(size)
plot_kernel_3d(hanning_kernel, "Normalizirano Hanningovo jedro")

# Generate and plot 2D Gaussian kernel
gaussian_kernel = gaussian_2d(size, sigma)
plot_kernel_3d(gaussian_kernel, "Normalizirano Gaussovo jedro")
