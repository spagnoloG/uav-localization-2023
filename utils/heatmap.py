import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

map_size = (100, 100)
point = (50, 50)
heatmap = np.zeros(map_size)
std_dev = 2
x, y = np.indices(map_size)
distance_from_point = np.sqrt((x - point[0])**2 + (y - point[1])**2)
heatmap = np.exp(-(distance_from_point**2 / (2.0 * std_dev**2)))
colors = ["blue",  "green",  "yellow", "orange", "red"]
cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)
plt.imshow(heatmap, cmap=cmap, interpolation='nearest')
plt.colorbar()
plt.show()
