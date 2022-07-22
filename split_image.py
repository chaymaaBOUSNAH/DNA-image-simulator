from PIL import Image
import os 

import os


'''
from matplotlib import pyplot as plt
#im = Image.open('BSQ_B140_19.tif')
import image_slicer
im = image_slicer.slice('BSQ_B140_19.tif', 4)
print(im)
image_slicer.save_tiles(im, directory='./chunks/',\
                            prefix='slice', format='png')


def crop(infile,height,width):
    im = Image.open(infile)
    imgwidth, imgheight = im.size
    for i in range(imgheight//height):
        for j in range(imgwidth//width):
            box = (j*width, i*height, (j+1)*width, (i+1)*height)
            yield im.crop(box)

if __name__=='__main__':
    infile='./BSQ_B140_19.tif'
    height=128
    width=128
    start_num=1
    for k,piece in enumerate(crop(infile,height,width),start_num):
        img=Image.new('RGB', (height,width), 255)
        img.paste(piece)
        path=os.path.join('./',"IMG-%s.png" % k)
        img.save(path)
'''            
from matplotlib import pyplot as plt
import tifffile
import numpy as np


img = tifffile.imread('./scanner_data/2022-06-21_10h41m25s_p100.0-100.0-100.0-g49-48-50_z120.0µm_Auto_20ls.tif')
img = img.transpose((1, 2, 0))
r = img[:, :, 2]
g = img[:, :, 1]
b = img[:, :, 0]

img = np.dstack((r, g, b))

img_h, img_w, _ = img.shape
split_width = 2048
split_height = 2048


def start_points(size, split_size, overlap=0):
    # définir le premier point de début de découpage qui est 0
    points = [0]
    # normalement chaque morceau doit prendre 1024 de l'image (split size) 
    # ms vu qu'il ya l'intersection entre les morceaux donc le 2ème point de découpage 
    # doit commencer à split_size- overlapsize = split_size- split_size*ovelap(%)  
    stride = int(split_size * (1-overlap))
    
    counter = 1
    while True:
        # pt est le point de début de chaque morceau
        pt = stride * counter
     
        if pt + split_size >= size:
          
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
 
    return points


X_points = start_points(img_w, split_width,  0.1)
print('X_points', X_points)
Y_points = start_points(img_h, split_height, 0.1)
print('Y_points',Y_points)
count = 0
name = 'splitted'
frmt = 'jpeg'

for i in Y_points:
    for j in X_points:
        
        #print('i', i)
        #print('i+split_height', i+split_height)
        
        #print('j', j)
        #print('j+split_width', j+split_width)
        
        split = img[i:i+split_height, j:j+split_width]
        
        #print('im', count, split.shape)
        plt.imshow(split)
        plt.show()
        
        split = np.uint8(np.clip(split, 0, 255))
        split = Image.fromarray(split)
        split.save('./chunks/chunk_'+str(count)+'.png')
        count += 1 
        