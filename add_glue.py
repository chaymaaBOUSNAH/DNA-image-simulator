from PIL import Image 
import matplotlib.pyplot as plt 
import numpy as np
import math  
import tifffile
import random
import pandas as pd
import os
#import cv2
import matplotlib as mpl
from os import walk
from os.path import join
from matplotlib import image
mpl.rc('figure', max_open_warning = 0)
import re 

def sorted_file( l ): 
    
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)
    


images_path = './sm_with_coord/essai/noisy_curves/'
for (dirpath, dirnames, filenames) in walk(images_path):
    
    file_index = 0
    for image_file in sorted_file(filenames):
        
        image_path = join(images_path, image_file)
        Imag = Image.open(image_path) 
        Imagecopy = Imag.copy() 
        
        dir_path = './sm_with_coord/glue/'
        for (dirpath, dirnames, file_names) in walk(dir_path):
            # n est le nombre aléatoire d'image de glue qui seront ajouter
            add_glue = ['true', 'false']
            adding_glue = random.choices(add_glue, weights=[0.9, 0.1])
            if adding_glue==['true']:
                print('true')
                n = np.random.randint(0, 6)
                random_glue = random.sample(file_names, n)
                for glue in random_glue:
                    
                    glue_size = np.random.randint(10, 150) 
                    glue_path = join(dir_path, glue)
                    glue_img = Image.open(glue_path)
                    
                    #angle = random.choice([0, 90, 180])
                    #glue_img = glue_img.rotate(angle)
                    gluecopy = glue_img.copy() 
                    
                    a = np.random.randint(0, 2048-glue_size)
                    b = np.random.randint(0, 2048-glue_size)
                    
                    Imagecopy.paste(gluecopy.resize((glue_size, glue_size)), (a, b)) 
                
                
        Imagecopy.save('./sm_with_coord/essai/noisy_glue/image_'+str(file_index)+'.png')       
        
        file_index +=1
'''
import matplotlib.pyplot as plt

img= plt.imread('./sm_with_coord/ess.png')


 
resolution = 10 #number of points used to draw the circle 
a = 10. #I arbitrary chose a center-x coordinate called a 
b = -5. #the same for y 
r = 12. #radius of the circle 
 
#x = [2*r*value/resolution + (a-r) for value in range(resolution-1)] 
#x += [a+r] #so that the last point is exactly at the right-end of the circle 
x = np.linspace(-12, 24, 100)
#x += [a+r]
 
y_plus = [(r**2 - (value-a)**2)**0.5 + b for value in x] 
y_minus = [2*b-value for value in y_plus] #y_plus - a = a - y_minus 
 
plt.figure(figsize=(6,6)) #so that the aspect ratio of circle is respected 
plt.plot(y_plus, x) 
#plt.plot(x,y_minus) 
plt.plot([a],[b],"x") #draw a cross at the center of the circle 
plt.show()

P1 = (1, 100)
P2 = (100,0)

def draw_cercle(P1, P2):
    x1, y1 = P1[0], P1[1]
    x2, y2 = P2[0], P2[1]
    diameter = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
    r = diameter / 2
    xC, yC = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    
    x = np.linspace(x1, x2, 100)
    
    y = [(r**2 - (value-xC)**2)**0.5 + yC for value in x] 
    
    return x, y 

x, y = draw_cercle(P1, P2)

plt.figure() #so that the aspect ratio of circle is respected 


plt.plot(x, y) 

plt.show()



from os import listdir
from os.path import isfile, join
from PIL import Image

from os import walk
# change 4 channel to 3 conversion from CMYK to RBG

images_path = './sm_with_coord/essai/train_mask'
for (dirpath, dirnames, filenames) in walk(images_path):

    for image_file in filenames:
        image_path = join(images_path, image_file)

        # load image with Pillow as RGB
        image = Image.open(image_path).convert("RGB")
        #image = np.asarray(image)
        #print('im shape', image.shape)
        r, g, b = image.getpixel((1, 1))

        print(r, g, b)
        #plt.imsave('./sm_with_coord/essai/train_mask/'+image_file, image, format='png')
        

    

import torch

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Create dummy target image
nb_classes = 19 - 1 # 18 classes + background
idx = np.linspace(0., 1., nb_classes)
cmap = matplotlib.cm.get_cmap('viridis')
rgb = cmap(idx, bytes=True)[:, :3]  # Remove alpha value

h, w = 190, 100
rgb = rgb.repeat(1000, 0)
target = np.zeros((h*w, 3), dtype=np.uint8)
target[:rgb.shape[0]] = rgb
target = target.reshape(h, w, 3)

plt.imshow(target) # Each class in 10 rows

# Create mapping
# Get color codes for dataset (maybe you would have to use more than a single
# image, if it doesn't contain all classes)
target = torch.from_numpy(target)
colors = torch.unique(target.view(-1, target.size(2)), dim=0).numpy()
target = target.permute(2, 0, 1).contiguous()

mapping = {tuple(c): t for c, t in zip(colors.tolist(), range(len(colors)))}

mask = torch.empty(h, w, dtype=torch.long)
for k in mapping:
    # Get all indices for current class
    idx = (target==torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
    validx = (idx.sum(0) == 3)  # Check that all channels match
    mask[validx] = torch.tensor(mapping[k], dtype=torch.long)        
'''