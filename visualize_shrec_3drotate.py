import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from data.input_pipeline import get_cshrec11

CSHREC11_DIR = "./data/CSHREC_11/processed/"
train_data, test_data, num_classes, (mean, std) = get_cshrec11(CSHREC11_DIR)

NUM_SHAPES = 8
for batch in train_data.batch(NUM_SHAPES).take(1):
    for key in ['image', 'shape', 'data']:
        if key in batch:
            data = batch[key].numpy()
            break
    else:
        data = list(batch.values())[0].numpy()
    break

if data.ndim == 5:
    data = data[0]  # (num_augs, H, W, num_channels)

num_augs, H, W, num_channels = data.shape

# Pick first augmentation and first channel
x = data[0][..., 0]  # (H, W)

# Spherical coordinates grid
theta = np.linspace(0, np.pi, H)  # latitude: 0 (north pole) to pi (south pole)
phi = np.linspace(0, 2 * np.pi, W)  # longitude: 0 to 2pi
phi_grid, theta_grid = np.meshgrid(phi, theta)

# Convert to 3D Cartesian coordinates
X = np.sin(theta_grid) * np.cos(phi_grid)
Y = np.sin(theta_grid) * np.sin(phi_grid)
Z = np.cos(theta_grid)

# Rotation function
def rotate_z(x, angle_deg):
    shift = int(W * angle_deg / 360)
    return np.roll(x, shift=shift, axis=1)

angles = [0, 90, 180, 270]
x_rots = [rotate_z(x, a) for a in angles]

fig = plt.figure(figsize=(18, 5))
for i, (a, x_r) in enumerate(zip(angles, x_rots)):
    ax = fig.add_subplot(1, 4, i+1, projection='3d')
    surf = ax.plot_surface(X, Y, Z, facecolors=plt.cm.viridis((x_r-x_r.min())/(x_r.max()-x_r.min())), rstride=1, cstride=1, antialiased=False, shade=False)
    ax.set_title(f'Rotated {a}\u00b0')
    ax.set_axis_off()
plt.tight_layout()
plt.show()
