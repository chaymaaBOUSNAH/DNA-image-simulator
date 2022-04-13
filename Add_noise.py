import numpy as np
import matplotlib.pyplot as plt
import math  
import tifffile
import random
import pandas as pd
import numpy as np
import os
import matplotlib as mpl
from PIL import Image 
from os import walk
from os.path import join
from matplotlib import cm
from scipy.signal import convolve2d


mpl.rc('figure', max_open_warning = 0)


import re 

def sorted_file( l ): 
    
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


N_Biologic_noise = np.random.randint(5, 40) # le nombre possible de fibres(bruit) dans une image
noisy_fibers = np.random.randint(5, 10)
noisy_dust = np.random.randint(5, 30)
m = np.random.uniform(0, 0.01)

images_path = './sm_with_coord/essai/noisy_glue/'
for (dirpath, dirnames, filenames) in walk(images_path):
    
    file_index = 0
    for image_file in sorted_file(filenames):
        
        image_path = join(images_path, image_file)
        
        imag = plt.imread(image_path)

        #output_img3 = convolver_rgb(output_img2, gaussian, 1)
        
        image = imag[:,:, 0:3]
               
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        fig.set_size_inches(2048/100, 2048/100)
        
        ax.imshow(image, cmap=plt.cm.gray, aspect='auto')       
    
        
        for j in range(noisy_fibers):
            
            #bruit bilogique: fibres d'ADN et analogues
            p1 =  np.random.uniform(1, 20)       
            # coordonnées x des fibres d'ADN
            x1 = np.random.uniform(0, 2048) 
            #lmin pour ne pas avoir des fibre trop petites(qui ressemblent au bruit) 
            x2 = x1+p1
            # coordonnées y des fibres d'ADN
            y1 = np.random.uniform(0, 2048) 
            #déterminer l'intercept de la droite
            b = y1 - m*x1
            # calculer y2 de la meme fibre
            y2 = m*x2 + b
            
            # morceaux des fibres des analogues comme bruit
            noise_colors = ['b', 'aqua', 'magenta']
            noise_color = np.random.choice(noise_colors, 1, p = [0.8, 0.1, 0.1])
            linewidth = np.random.randint(2, 8)
            plt.plot((x1, x2),(y1, y2), color= noise_color[0], linewidth=linewidth)
            
        for noise in range(N_Biologic_noise):     
            a = np.random.uniform(0, 2048)
            b = np.random.uniform(0, 2048)
            s = np.random.randint(1, 20)
            plt.scatter(a, b, marker='o', c = 'b', s = s)
            
        for noisy_dust in range(noisy_dust):   
            # autre bruit : poussière
            x = np.random.uniform(0, 2048)
            y = np.random.uniform(0, 2048)

            
            colors = ['g', 'r']
            color = random.choices(colors, weights=[0.6, 0.4])
            alpha_value = 0.3
            
            n_point =  np.random.randint(10, 50)
            s = np.random.randint(1, 30)
                
            for n in range(1, n_point+1):
                if n<5:
                    plt.scatter(x,y, marker='o', c = 'w', s = s*n, alpha = alpha_value/n)
                else:
                    plt.scatter(x,y, marker='o',color = color, s = s*n , alpha = alpha_value/(n*1.5))
                    
                        

            #plt.scatter(x,y, marker='o', s=w , alpha = alpha_value, color = color, cmap = viridis)
            
        ax.set_xlim((0, image.shape[1]))
        ax.set_ylim((image.shape[0], 0))
        #ax[1].set_title('draw lines')
        
               
        plt.savefig('./sm_with_coord/essai/noisy_fibers/'+image_file, bbox_inches='tight', pad_inches=0, dpi = 100)
        print('image saved')
        
        
        file_index +=1



# Sharpen
sharpen = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])
# Gaussian Blur
gaussian = (1 / 16.0) * np.array([[1., 2., 1.],
                                  [2., 4., 2.],
                                  [1., 2., 1.]])
'''
fig, ax = plt.subplots(1,2, figsize = (17,10))
ax[0].imshow(sharpen, cmap='gray')
ax[0].set_title('Sharpen', fontsize = 18)
    
ax[1].imshow(gaussian, cmap='gray')
ax[1].set_title('Gaussian Blur', fontsize = 18)
    
[axi.set_axis_off() for axi in ax.ravel()];
'''

def multi_convolver(image, kernel, iterations):
    for i in range(iterations):
        image = convolve2d(image, kernel, 'same', boundary = 'fill',
                           fillvalue = 0)
    return image

def convolver_rgb(image, kernel, iterations = 1):
    convolved_image_r = multi_convolver(image[:,:,0], kernel,
                                        iterations)
    convolved_image_g = multi_convolver(image[:,:,1], kernel, 
                                        iterations)
    convolved_image_b  = multi_convolver(image[:,:,2], kernel, 
                                         iterations)
    
    reformed_image = np.dstack((np.rint(abs(convolved_image_r)), 
                                np.rint(abs(convolved_image_g)), 
                                np.rint(abs(convolved_image_b)))) 

    return reformed_image


'''
Parameters
----------
image : ndarray
    Input image data. Will be converted to float.
mode : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    's&p'       Replaces random pixels with 0 or 1.

''' 

def noisy(noise_typ,image):
    
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0# Mean (“centre”) of the distribution.
      #var = np.random.uniform(0.1, 0.3)  # 0.3
      #sigma = np.random.uniform(0.5, 1.5)  # Standard deviation (spread or “width”) of the distribution.
      #print('sigma', sigma)
      sigma = np.random.uniform(0.2, 1)
      "Draw random samples from a normal (Gaussian) distribution."
      # row * col * ch samples are drawn
      gauss = np.random.normal(mean,sigma,(row,col,ch)) 
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
  
   elif noise_typ == "s&p":
      row,col,ch = image.shape
      # s_vs_p is the pourcentage of salt     
      s_vs_p = np.random.uniform(0, 0.3) 
      # pourcentage of noise to add
      amount =  np.random.uniform(0.1, 0.8) 
      out = np.copy(image)
      
      # Add Salt 
      # ceil return the smallest integer of a float 
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      #coords = np.array(coords)
      out[coords] = 1

      # Add Pepper 
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      #coords = np.array(coords)
      out[coords] = 0
            
      return out
  
"""
def sp_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = image.copy()
    if len(image.shape) == 2:
        black = 0
        white = 255            
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    return output

"""
# salt & pepper noise for rgb image
def sp_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = image.copy()
    
    black = np.array([0, 0, 0], dtype='uint8')
    white = np.array([255, 255, 255], dtype='uint8')
       
    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    return output

def s_p_noise(image, prob):
    output = image.copy()
    blue_channel = output[:, :, 2]
    red_channel = output[:, :, 0]
    green_channel = output[:, :, 1]
    
    channels = [red_channel, green_channel, blue_channel]
    noisy_image = []
    for channel in channels:
        s_vs_p = np.random.uniform(0, 0.5) 
        # pourcentage of noise to add
        amount =  np.random.uniform(0.1, 0.8) 
    
        # Add Salt 
        # ceil return the smallest integer of a float 
        num_salt = np.ceil(amount * channel.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in channel.shape]
        #coords = np.array(coords)
        channel[coords] = 1
        
        # Add Pepper 
        num_pepper = np.ceil(amount* channel.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in channel.shape]
        #coords = np.array(coords)
        channel[coords] = 0
        
        noisy_image.append()
        
            
    

cmap = cm.get_cmap("Spectral")
viridis = cm.get_cmap('viridis')

N_Biologic_noise = np.random.randint(20, 80) # le nombre possible de fibres(bruit) dans une image


images_path = './sm_with_coord/essai/noisy_fibers/'
for (dirpath, dirnames, filenames) in walk(images_path):
    
    file_index = 0
    for image_file in sorted_file(filenames):
        
        image_path = join(images_path, image_file)
        '''
        imag = Image.open(image_path).convert('RGB')
        imag = np.array(imag)
        '''
        imag = plt.imread(image_path)
        imag = imag[:, :, :3]
          
        output_img1 = noisy("gauss", imag)
        # Image = Image/np.amax(Image)
        output_img1 = np.clip(output_img1, 0, 1)
        
        output_img2 = noisy("s&p", output_img1)

        #output_img3 = convolver_rgb(output_img2, gaussian, 1)
        '''
        First ensure your NumPy array, myarray, is normalised with the max value at 1.0.
        Apply the colormap directly to myarray.
        Rescale to the 0-255 range.
        Convert to integers, using np.uint8().
        Use Image.fromarray().
        '''
        #output_img2 = Image.fromarray(np.uint8(output_img2))
        
        plt.imsave('./sm_with_coord/essai/noisy2/'+image_file, output_img2)
        file_index +=1

