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

# Select the first sample in the batch
x = data[0]  # shape: (96, 192, 16)

# Apply a 90-degree rotation (shift by 1/4 along the longitude axis)
def rotate_longitude_90(x):
    shift = x.shape[1] // 4
    return np.roll(x, shift=shift, axis=1)

x_rot = rotate_longitude_90(x)

# Visualize the first channel before and after rotation
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(x[..., 0], cmap='gray')
axes[0].set_title('Original (channel 0)')
axes[0].axis('off')
axes[1].imshow(x_rot[..., 0], cmap='gray')
axes[1].set_title('90° Rotated (channel 0)')
axes[1].axis('off')
plt.tight_layout()
plt.show()
