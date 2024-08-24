import numpy as np
import matplotlib.pyplot as plt
import random

data = np.load('data.npy')
len = len(data)
sel = random.sample(range(len), 9)

# Set colormap to gray
plt.gray()

# Plot images
fig, axs = plt.subplots(3, 3, figsize=(10,10))
for ax in axs.ravel():
    ax.set_axis_off()
axs[0, 0].imshow(data[sel[0]])
axs[0, 1].imshow(data[sel[1]])
axs[0, 2].imshow(data[sel[2]])
axs[1, 0].imshow(data[sel[3]])
axs[1, 1].imshow(data[sel[4]])
axs[1, 2].imshow(data[sel[5]])
axs[2, 0].imshow(data[sel[6]])
axs[2, 1].imshow(data[sel[7]])
axs[2, 2].imshow(data[sel[8]])
fig.tight_layout()
plt.show()