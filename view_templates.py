import numpy as np
import matplotlib.pyplot as plt

data = np.load('templates.npy')
circle_bar = data[0]
circle_cross = data[1]
circle_star = data[2]
square_bar = data[3]
square_cross = data[4]
square_star = data[5]
triangle_bar = data[6]
triangle_cross = data[7]
triangle_star = data[8]

# Set colormap to gray
plt.gray()

# Plot images
fig, axs = plt.subplots(3, 3, figsize=(10,10))
for ax in axs.ravel():
    ax.set_axis_off()
axs[0, 0].imshow(circle_bar)
axs[0, 1].imshow(circle_cross)
axs[0, 2].imshow(circle_star)
axs[1, 0].imshow(square_bar)
axs[1, 1].imshow(square_cross)
axs[1, 2].imshow(square_star)
axs[2, 0].imshow(triangle_bar)
axs[2, 1].imshow(triangle_cross)
axs[2, 2].imshow(triangle_star)
fig.tight_layout()
plt.show()