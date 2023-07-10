import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Correlation(nn.Module):
    def __init__(self):
        super(Correlation, self).__init__()
        self.batch_norm = nn.BatchNorm2d(1)

    def forward(self, query, search_map):
        # Ensure the input tensors are 4D (batch, channels, height, width)
        assert query.dim() == search_map.dim() == 4

        # Group convolution as correlation
        # Pad search map to maintain spatial resolution
        search_map_padded = F.pad(
            search_map,
            (
                query.shape[3] // 2,
                query.shape[3] // 2,
                query.shape[2] // 2,
                query.shape[2] // 2,
            ),
        )

        # Get shape for convenience
        bs, c, h, w = query.shape
        _, _, H, W = search_map_padded.shape

        corr_maps = []
        for map_, q_ in zip(search_map_padded, query):
            map_ = map_.view(1, c, H, W)
            q_ = q_.view(1, c, h, w)

            corr_map = F.conv2d(map_, q_, groups=1)
            corr_maps.append(corr_map)

        corr_maps = torch.cat(corr_maps, dim=0)
        # corr_maps = self.batch_norm(corr_maps)

        return corr_maps


corr_module = Correlation()

sat_features = torch.zeros(4, 320, 32, 32)
drone_features = torch.zeros(4, 320, 16, 16)

# Add some structure to the features
sat_features[:, 0, :16, :16] = 1.0
drone_features[:, 0, 8:12, 8:12] = 1.0

# Apply match_corr
# corr_map = match_corr(drone_features, sat_features)
corr_map = corr_module(drone_features, sat_features)

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

axs[0].imshow(sat_features[0, 0], cmap="gray")
axs[0].set_title("Satellite feature (channel 0)")

axs[1].imshow(drone_features[0, 0], cmap="gray")
axs[1].set_title("Drone feature (channel 0)")

axs[2].imshow(corr_map[0, 0], cmap="gray")
axs[2].set_title("Correlation map (channel 0)")

plt.tight_layout()
plt.show()
print(corr_map.shape)
# Print the values of channel 0 of the correlation map
print(corr_map[0, 0])
