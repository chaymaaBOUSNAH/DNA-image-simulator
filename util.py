import os.path
import re
import math
from PIL import Image
import numpy as np


def create_directory(dir_path):
    if os.path.isdir(dir_path)=='':
        print('path is empty, enter a valid path')
    if os.path.isdir(dir_path):
        print('directory already exists !')
    else:
        os.makedirs(dir_path)
        print('directory is created !')


def Read_Image_to_numpy_arr(image_path):
    if image_path=='':
        print('the input image path is empty')
    else:
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        return image


def sorted_file( l ):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


def distance(p1, p2):
    dist = math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    return dist