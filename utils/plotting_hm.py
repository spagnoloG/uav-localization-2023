import torch
import matplotlib.pyplot as plt

# Define target tensor
target = torch.zeros(1, 400, 400)
target[:, 100:133, 100:133] = 1

# Plot target tensor
plt.imshow(target[0], cmap="gray")
plt.colorbar()
plt.title("Target Tensor")
plt.savefig("Target Tensor.jpg", format="jpg")
plt.show()
