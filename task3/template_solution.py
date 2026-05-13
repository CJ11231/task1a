from pathlib import Path

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

"""
README FIRST

The below code is a template for the solution. You can change the code according
to your preferences, but the testing function has to save the output of your
model on the test data as it does in this template. This output must be submitted.

Replace the dummy code with your own code in the TODO sections.

We also encourage you to use tensorboard or wandb to log the training process
and the performance of your model. This will help you to debug your model and
to understand how it is performing. But the template does not include this
functionality.
Link for wandb:
https://docs.wandb.ai/quickstart/
Link for tensorboard:
https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
"""

# The device is automatically set to GPU if available, otherwise CPU
# If you want to force the device to CPU, you can change the line to
# device = torch.device("cpu")

# If you have a Mac consult the following link:
# https://pytorch.org/docs/stable/notes/mps.html

# It is important that your model and all data are on the same device.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data(**kwargs):
    """
    Get the training and test data. The data files are assumed to be in the
    same directory as this script.

    Args:
    - kwargs: Additional arguments that you might find useful - not necessary

    Returns:
    - train_data_input: Tensor[N_train_samples, C, H, W]
    - train_data_label: Tensor[N_train_samples, C, H, W]
    - test_data_input: Tensor[N_test_samples, C, H, W]
    where N_train_samples is the number of training samples, N_test_samples is
    the number of test samples, C is the number of channels (1 for grayscale),
    H is the height of the image, and W is the width of the image.
    """
    # Load the training data
    train_data = np.load("train_data.npz")["data"]

    # Make the training data a tensor
    train_data = torch.tensor(train_data, dtype=torch.float32)

    # Load the test data
    test_data_input = np.load("test_data.npz")["data"]

    # Make the test data a tensor
    test_data_input = torch.tensor(test_data_input, dtype=torch.float32)

    # Labels are the original (complete) images
    train_data_label = train_data.clone()

    # Inputs are the masked images: center 8x8 pixels set to 0
    train_data_input = train_data.clone()
    train_data_input[:, :, 10:18, 10:18] = 0.0

    # Visualize the training data if needed
    # Set to False if you don't want to save the images
    if True:
        # Create the output directory if it doesn't exist
        if not Path("train_image_output").exists():
            Path("train_image_output").mkdir()
        for i in tqdm(range(20), desc="Plotting train images"):
            # Show the training and the target image side by side
            plt.subplot(1, 2, 1)
            plt.imshow(train_data_input[i].squeeze(), cmap="gray")
            plt.title("Training Input")
            plt.subplot(1, 2, 2)
            plt.title("Training Label")
            plt.imshow(train_data_label[i].squeeze(), cmap="gray")

            plt.savefig(f"train_image_output/image_{i}.png")
            plt.close()

    return train_data_input, train_data_label, test_data_input


def training(train_data_input, train_data_label, **kwargs):
    """
    Train the model. Fill in the details of the data loader, the loss function,
    the optimizer, and the training loop.

    Args:
    - train_data_input: Tensor[N_train_samples, C, H, W]
    - train_data_label: Tensor[N_train_samples, C, H, W]
    - kwargs: Additional arguments that you might find useful - not necessary

    Returns:
    - model: torch.nn.Module
    """
    # Normalize to [0, 1] for training
    train_data_input = train_data_input / 255.0
    train_data_label = train_data_label / 255.0

    model = Model()
    model.train()
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 128
    dataset = TensorDataset(train_data_input, train_data_label)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    n_epochs = 30
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for x, y in tqdm(
            data_loader, desc=f"Training Epoch {epoch}", leave=False
        ):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss_full = criterion(output, y)
            loss_masked = criterion(output[:, :, 10:18, 10:18], y[:, :, 10:18, 10:18])
            loss = loss_full + 9.0 * loss_masked
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch} loss: {epoch_loss / len(data_loader):.6f}")

    return model


class Model(nn.Module):
    """
    A U-Net-like convolutional autoencoder for image inpainting.
    This architecture uses skip connections to preserve high-resolution details,
    which is beneficial for reconstruction tasks. It uses BatchNorm for stability.
    """

    def __init__(self):
        super().__init__()

        # Encoder (contracting path)
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Decoder (expansive path)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # Output layer
        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc_conv1(x)
        pool1 = self.pool1(enc1)
        enc2 = self.enc_conv2(pool1)
        pool2 = self.pool2(enc2)

        # Bottleneck
        bottle = self.bottleneck(pool2)

        # Decoder with skip connections
        up1 = self.upconv1(bottle)
        cat1 = torch.cat([up1, enc2], dim=1)
        dec1 = self.dec_conv1(cat1)

        up2 = self.upconv2(dec1)
        cat2 = torch.cat([up2, enc1], dim=1)
        dec2 = self.dec_conv2(cat2)

        return self.out_conv(dec2)


def testing(model, test_data_input):
    """
    Uses your model to predict the ouputs for the test data. Saves the outputs
    as a binary file. This file needs to be submitted. This function does not
    need to be modified except for setting the batch_size value. If you choose
    to modify it otherwise, please ensure that the generating and saving of the
    output data is not modified.

    Args:
    - model: torch.nn.Module
    - test_data_input: Tensor
    """
    model.eval()
    model.to(device)

    # Normalize test input the same way as training
    test_data_input_norm = test_data_input / 255.0

    with torch.no_grad():
        test_data_input_norm = test_data_input_norm.to(device)
        test_data_output = []
        batch_size = 64
        for i in tqdm(
            range(0, test_data_input_norm.shape[0], batch_size),
            desc="Predicting test output",
        ):
            output = model(test_data_input_norm[i : i + batch_size])
            # Scale back to [0, 255]
            output = output * 255.0
            test_data_output.append(output.cpu())
        test_data_output = torch.cat(test_data_output)

    # Ensure the output has the correct shape
    assert test_data_output.shape == test_data_input.shape, (
        f"Expected shape {test_data_input.shape}, but got "
        f"{test_data_output.shape}."
        "Please ensure the output has the correct shape."
        "Without the correct shape, the submission cannot be evaluated and "
        "will hence not be valid."
    )

    # Save the output
    test_data_output = test_data_output.numpy()
    # Ensure all values are in the range [0, 255]
    save_data_clipped = np.clip(test_data_output, 0, 255)
    # Convert to uint8
    # Ensure your model outputs values in the [0, 255] range before this step! If you normalized your data to [0, 1], you must multiply by 255 before saving.
    save_data_uint8 = save_data_clipped.astype(np.uint8)
    # Loss is only computed on the masked area - so set the rest to 0 to save
    # space
    save_data = np.zeros_like(save_data_uint8)
    save_data[:, :, 10:18, 10:18] = save_data_uint8[:, :, 10:18, 10:18]

    np.savez_compressed(
        "submit_this_test_data_output.npz", data=save_data)

    # You can plot the output if you want
    # Set to False if you don't want to save the images
    if True:
        # Create the output directory if it doesn't exist
        if not Path("test_image_output").exists():
            Path("test_image_output").mkdir()
        for i in tqdm(range(20), desc="Plotting test images"):
            # Show the training and the target image side by side
            plt.subplot(1, 2, 1)
            plt.title("Test Input")
            plt.imshow(test_data_input[i].squeeze().cpu().numpy(), cmap="gray")
            plt.subplot(1, 2, 2)
            plt.imshow(test_data_output[i].squeeze(), cmap="gray")
            plt.title("Test Output")

            plt.savefig(f"test_image_output/image_{i}.png")
            plt.close()


def main():
    seed = 0
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # You don't need to change the code below
    # Load the data
    train_data_input, train_data_label, test_data_input = load_data()
    # Train the model
    model = training(train_data_input, train_data_label)

    # Test the model (this also generates the submission file)
    # The name of the submission file is submit_this_test_data_output.npz
    testing(model, test_data_input)

    return None


if __name__ == "__main__":
    main()
