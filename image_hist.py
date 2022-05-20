import matplotlib.pyplot as plt
import numpy as np


image_reelle = plt.imread('./BSQ_B140_19.tif')
image_generateur = plt.imread('./create_images/output_images/image_0.png')
image_generateur = (image_generateur*255).astype(np.uint8)

# tuple to select colors of each channel line
colors = ("red", "green", "blue")
channel_ids = (0, 1, 2)

# create the histogram plot, with three lines, one for
# each color
plt.figure()
plt.xlim([0, 256])

for channel_id, c in zip(channel_ids, colors):
    histogram, bin_edges = np.histogram(
        image_generateur[:, :, channel_id], bins=256, range=(0, 256)
    )
    
    plt.plot(bin_edges[0:-1], histogram, color=c)
    
    
    


plt.title("Color Histogram")
plt.xlabel("Color value")
plt.ylabel("Pixel count")

plt.show()

