import torch
import torch.optim as optim
from model import CustomResNetDeiT
from model import BalanceLoss

# Assuming you have the DataLoader objects `train_loader` and `val_loader`

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = CustomResNetDeiT().to(device)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the loss function
loss_fn = BalanceLoss()

# Number of training epochs
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    train_loss = 0

    for drone_imgs, satellite_imgs, labels in train_loader:
        # Move the images and labels to the GPU if available
        drone_imgs = drone_imgs.to(device)
        satellite_imgs = satellite_imgs.to(device)
        labels = labels.to(device)

        # Forward propagation
        outputs = model(drone_imgs, satellite_imgs)

        # Calculate the loss
        loss = loss_fn(outputs, labels)

        # Backward propagation and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * drone_imgs.size(0)

    # Calculate average losses
    train_loss = train_loss / len(train_loader.dataset)

    # Print loss statistics
    print("Epoch: {}/{}, Train Loss: {:.4f}".format(epoch + 1, num_epochs, train_loss))

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        val_loss = 0
        for drone_imgs, satellite_imgs, labels in val_loader:
            drone_imgs = drone_imgs.to(device)
            satellite_imgs = satellite_imgs.to(device)
            labels = labels.to(device)

            outputs = model(drone_imgs, satellite_imgs)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item() * drone_imgs.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        print("Val Loss: {:.4f}".format(val_loss))
