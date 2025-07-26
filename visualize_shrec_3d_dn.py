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
theta = np.linspace(0, np.pi, H)  # latitude
phi = np.linspace(0, 2 * np.pi, W)  # longitude
phi_grid, theta_grid = np.meshgrid(phi, theta)

# Convert to 3D Cartesian coordinates
X = np.sin(theta_grid) * np.cos(phi_grid)
Y = np.sin(theta_grid) * np.sin(phi_grid)
Z = np.cos(theta_grid)

# D_n symmetries (n = 4 for 0, 90, 180, 270)
def dn_symmetries(x, n):
    H, W = x.shape
    syms = []
    for k in range(n):
        # Rotation
        rot = np.roll(x, shift=k*W//n, axis=1)
        syms.append((f'rot_{k*360//n}', rot))
        # Reflection + rotation
        refl = np.flip(rot, axis=0)
        syms.append((f'reflect_rot_{k*360//n}', refl))
    return syms

n = 4
syms = dn_symmetries(x, n)

fig = plt.figure(figsize=(18, 9))
for i, (name, x_r) in enumerate(syms):
    ax = fig.add_subplot(2, n, i+1, projection='3d')
    surf = ax.plot_surface(X, Y, Z, facecolors=plt.cm.viridis((x_r-x_r.min())/(x_r.max()-x_r.min())), rstride=1, cstride=1, antialiased=False, shade=False)
    # Mark north pole (0,0,1) and south pole (0,0,-1)
    ax.scatter([0], [0], [1], color='red', s=60, label='North Pole')
    ax.scatter([0], [0], [-1], color='blue', s=60, label='South Pole')
    ax.text(0, 0, 1.1, 'N', color='red', fontsize=14, ha='center', va='center', weight='bold')
    ax.text(0, 0, -1.1, 'S', color='blue', fontsize=14, ha='center', va='center', weight='bold')
    ax.set_title(name)
    ax.set_axis_off()
    ax.view_init(elev=0, azim=0)  # Side view (equator edge-on)
plt.tight_layout()
plt.show()
