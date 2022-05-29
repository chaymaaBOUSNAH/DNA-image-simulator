import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# read image
path = './image_essai.png'
image_path = './output_images/image_20.png'
image = Image.open(image_path).convert('RGBA')
img = np.array(image)

ht, wd = img.shape[:2]



red_channel = img[:, :, 0]
green_channel = img[:, :, 1]
blue_channel = img[:, :, 2]



# compute 5% of ht and 95% of ht
# pct = 5
pct = 25    # temparily set pct to 25 percent for demonstration
ht2 = int(ht*pct/100)
ht3 = ht - ht2

# create opaque white image for top
top = np.full((ht3,wd), 255, dtype=np.uint8)

# create vertical gradient for bottom
btm = np.linspace(255, 0, ht2, endpoint=True, dtype=np.uint8)
btm = np.tile(btm, (wd,1))
btm = np.transpose(btm)

# stack top and bottom
alpha = np.vstack((top,btm))

# put alpha channel into image
result = img.copy()

result[:,:,3] = alpha
result_img = Image.fromarray(result)

# display results
# (note: display does not show transparency)

result_img.show()
result_img.save('./image_essai.png')