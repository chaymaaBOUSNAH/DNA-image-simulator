import numpy as np
from PIL import Image



def get_gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T
    
    
def get_gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=np.float64)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = get_gradient_2d(start, stop, width, height, is_horizontal)

    return result

image_path = './output_images/image_20.png'
image = Image.open(image_path).convert('RGB')
img = np.array(image)

ht, wd = img.shape[:2]


red_channel = img[:, :, 0]
green_channel = img[:, :, 1]
blue_channel = img[:, :, 2]


array = get_gradient_2d(50, 0, ht, wd, True)

red_channel_ = red_channel - array
red_channel_ = np.uint8(np.clip(red_channel_, 0, 255))

green_channel_ = green_channel - array
green_channel_ = np.uint8(np.clip(green_channel_, 0, 255))


blue_channel_ = blue_channel- array
blue_channel_ = np.uint8(np.clip(blue_channel_, 0, 255))

noisy = np.dstack((red_channel_, green_channel_, blue_channel_))


array_rgb = get_gradient_3d(ht, wd, (30, 50, 20), (0, 0, 0), (True, False, False))
array_rgb = np.uint8(np.clip(array_rgb, 0, 255))

img_ = img - array_rgb

img_ = np.uint8(np.clip(img_, 0, 255))

im = Image.fromarray(img_)

im.save('./essai.png')