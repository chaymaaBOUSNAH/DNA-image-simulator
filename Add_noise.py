import numpy as np
import matplotlib.pyplot as plt
import math  
import tifffile
import random
import pandas as pd
import numpy as np
import os
#import cv2
import matplotlib as mpl

from os import walk
from os.path import join
from matplotlib.patches import Arc

from matplotlib import cm
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.filters import threshold_mean, threshold_triangle

from Bezier_curve import Bezier

mpl.rc('figure', max_open_warning = 0)


import re 

def sorted_file( l ): 
    
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)




'''
Parameters
----------
image : ndarray
    Input image data. Will be converted to float.
mode : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
    n is uniform noise with specified mean & variance.
''' 


def noisy(noise_typ,image):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0 # Mean (“centre”) of the distribution.
      var = np.random.uniform(0.1, 0.3)  # 0.3
      print('var', var)
      sigma = var**0.5 # Standard deviation (spread or “width”) of the distribution.
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
  
   elif noise_typ == "s&p":
      row,col,ch = image.shape
      print('image.shape', image.shape)
           
      s_vs_p = np.random.uniform(0.1, 0.2) 
      amount =  np.random.uniform(0.1, 0.5) 
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      print('num_salt', num_salt)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      #coords = np.array(coords)
      out[coords] = 255

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      print('num_pepper', num_pepper)
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      #coords = np.array(coords)
      out[coords] = 0
      return out
  
   elif noise_typ == "poisson":
       vals = len(np.unique(image))
       vals = 2 ** np.ceil(np.log2(vals))
       noisy = np.random.poisson(image * vals) / float(vals)
       return noisy
   elif noise_typ =="speckle":
       row,col,ch = image.shape
       gauss = np.random.randn(row,col,ch)
       gauss = gauss.reshape(row,col,ch)        
       noisy = image + image * gauss
       return noisy




cmap = cm.get_cmap("Spectral")
viridis = cm.get_cmap('viridis')

N_Biologic_noise = np.random.randint(20, 70) # le nombre possible de fibre dans une image


images_path = './sm_with_coord/essai/noisy_glue/'
for (dirpath, dirnames, filenames) in walk(images_path):
    
    file_index = 0
    for image_file in sorted_file(filenames):
        
        image_path = join(images_path, image_file)
        
        imag = plt.imread(image_path)
        
              
        output_img1 = noisy("gauss", imag)
        
        output_img2 = noisy("s&p", output_img1)
        #plt.imshow(output_img2 * 255).astype(np.uint8)
        #output_img3 = noisy("speckle", output_img2)
    
        image = output_img2[:,:, 0:3]
        
        #blurred = gaussian_filter(image, sigma=5)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        fig.set_size_inches(2048/100, 2048/100)
        
        ax.imshow(image, cmap=plt.cm.gray, aspect='auto')       
    
        
        for j in range(N_Biologic_noise):
            
            #bruit bilogique: fibres d'ADN et analogues
            a =  np.random.uniform(0, 2048)
            b =  np.random.uniform(0, 2048)
            p =  np.random.uniform(-10, 10)
            
            noise_colors = ['w', 'b', 'r', 'g']
            noise_color = random.choice(noise_colors)
            plt.plot((a, a+p),(b,b+p), color= noise_color)
            
            # autre bruit : poussière
            x = np.random.uniform(0, 2048)
            y = np.random.uniform(0, 2048)

            
            colors = ['g', 'r', 'pink']
            color = random.choices(colors, weights=[0.6, 0.3, 0.1])
            alpha_value = 0.3
            
            n_point =  np.random.randint(10, 60)
            s = np.random.randint(1, 20)
                
            for n in range(1, n_point+1):
                if n<5:
                    plt.scatter(x,y, marker='o', c = 'w', s = s*n, alpha = alpha_value/n)
                else:
                    plt.scatter(x,y, marker='o',color = color, s = s*n , alpha = alpha_value/(n*1.5))
                    
                        

            #plt.scatter(x,y, marker='o', s=w , alpha = alpha_value, color = color, cmap = viridis)
            
        ax.set_xlim((0, image.shape[1]))
        ax.set_ylim((image.shape[0], 0))
        #ax[1].set_title('draw lines')
        
               
        plt.savefig('./sm_with_coord/essai/noisy/'+image_file, bbox_inches='tight', pad_inches=0, dpi = 100)
        
        
        file_index +=1

