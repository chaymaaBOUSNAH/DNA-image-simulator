import math
import numpy as np
from pathlib import Path
import pandas as pd
import os

def canvas2rgb_array(canvas):
    
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    ncols, nrows = canvas.get_width_height()

    scale = round(math.sqrt(buf.size / 3 / nrows / ncols))
    return buf.reshape(scale * nrows, scale * ncols, 3)


def distance(p1, p2):
    dist = math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    return dist


def corresponding_csv_img(csv_path, image_file):
    # extact the coresponding csv file using the image name
    # extract image name without extention
    image_name = image_file.split('.')
    name = image_name[0]
    # extract csv path using image name
    csv_fibers = list(Path(csv_path).glob(name))[0]
    return csv_fibers

def verify_csv(csv_path, image_file):

    # if empty path
    if csv_path == '':
        return'No path found, enter a valid path to csv file !'
    else:
        csv_fibers = corresponding_csv_img(csv_path, image_file)
        file_fibers = os.path.basename(csv_fibers)

        if len([file_fibers]) == 0:
            return 'Np csv file found !'
        elif len([file_fibers]) > 1:
            return 'More than one csv file found for one image'
        else:
            return "csv data is extracted"




