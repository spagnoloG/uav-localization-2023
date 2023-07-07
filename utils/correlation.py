import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sat_features = torch.zeros(4, 320, 32, 32)
drone_features = torch.zeros(4, 320, 16, 16)


# Define match_corr function
def match_corr(drone_feature, satellite_feature):
    b, c, h, w = satellite_feature.shape

    match_map = F.conv2d(
        satellite_feature.view(1, b * c, h, w), drone_feature, groups=b
    )
    match_map = match_map.permute(1, 0, 2, 3)

    return match_map


sat_features = torch.zeros(4, 320, 32, 32)
drone_features = torch.zeros(4, 320, 16, 16)

# Add some structure to the features
sat_features[:, 0, :16, :16] = 1.0
drone_features[:, 0, 8:12, 8:12] = 1.0

# Apply match_corr
corr_map = match_corr(drone_features, sat_features)

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

axs[0].imshow(sat_features[0, 0].detach().numpy(), cmap="gray")
axs[0].set_title("Satellite feature (channel 0)")

axs[1].imshow(drone_features[0, 0].detach().numpy(), cmap="gray")
axs[1].set_title("Drone feature (channel 0)")

axs[2].imshow(corr_map[0, 0].detach().numpy(), cmap="gray")
axs[2].set_title("Correlation map (channel 0)")

plt.tight_layout()
plt.show()
