import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from data.input_pipeline import get_cshrec11

# Set the path to your SHREC11 dataset directory
CSHREC11_DIR = "./data/CSHREC_11/processed/"

# Load train data (tf.data.Dataset)
train_data, test_data, num_classes, (mean, std) = get_cshrec11(CSHREC11_DIR)

# Get a batch of real data
NUM_SHAPES = 8
for batch in train_data.batch(NUM_SHAPES).take(1):
    for key in ['image', 'shape', 'data']:
        if key in batch:
            data = batch[key].numpy()
            break
    else:
        data = list(batch.values())[0].numpy()
    break

num_augs = data.shape[0]
H, W, num_channels = data.shape[1:]

print(f"Data shape: {data.shape} (num_augs, H, W, num_channels)")

# Interactive selection
def visualize_aug_channel(aug_idx=0, channel_idx=0):
    x = data[aug_idx]  # (H, W, num_channels)
    x_rot = np.roll(x, shift=W//4, axis=1)  # 90 degree rotation
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(x[..., channel_idx], cmap='gray')
    axes[0].set_title(f'Original (aug={aug_idx}, ch={channel_idx})')
    axes[0].axis('off')
    axes[1].imshow(x_rot[..., channel_idx], cmap='gray')
    axes[1].set_title(f'90° Rotated (aug={aug_idx}, ch={channel_idx})')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print(f"Available augmentations: 0 to {num_augs-1}")
    print(f"Available channels: 0 to {num_channels-1}")
    aug_idx = int(input(f"Enter augmentation index [0-{num_augs-1}]: "))
    channel_idx = int(input(f"Enter channel index [0-{num_channels-1}]: "))
    visualize_aug_channel(aug_idx, channel_idx)
