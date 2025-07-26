import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from data.xforms import draw_shrec11_pairs

# Dummy batch: shape (batch_size, num_shapes, H, W, C)
batch_size = 4
num_shapes = 8
H, W, C = 64, 64, 1  # Example shape for SHREC11
data = np.random.rand(batch_size, num_shapes, H, W, C).astype(np.float32)

# JAX random key
test_key = jax.random.PRNGKey(0)

# Draw pairs
pairs = draw_shrec11_pairs(jnp.array(data), test_key)

# Visualize a few pairs (x, Tx)
num_pairs_to_show = 4
fig, axes = plt.subplots(num_pairs_to_show, 2, figsize=(5, 2 * num_pairs_to_show))
for i in range(num_pairs_to_show):
    x = pairs[2*i, ...].squeeze()
    Tx = pairs[2*i+1, ...].squeeze()
    axes[i, 0].imshow(x, cmap='gray')
    axes[i, 0].set_title('x')
    axes[i, 0].axis('off')
    axes[i, 1].imshow(Tx, cmap='gray')
    axes[i, 1].set_title('Tx')
    axes[i, 1].axis('off')
plt.tight_layout()
plt.show()
