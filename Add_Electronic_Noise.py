import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random
from Add_gaussian_2D_image import Gaussian_noise, Gaussian_noise_RGB, Add_Salt

def Add_Electronic_noise(image, amount_SP, sigma_green_ch, gaussian_Blur_sigma, Parasites_green_ch, Parasites_red_ch):
   
    row,col,ch= image.shape
    
    output = (image).astype(np.uint8)
    
    blue_channel = output[:, :, 2]
    red_channel = output[:, :, 0]
    green_channel = output[:, :, 1]
    
    channels = [red_channel, green_channel, blue_channel]
    noisy_image = []
    for i in range(ch):
        
        channel = output[: , :, i]
        
        if i ==0:
            channel = channel + Parasites_red_ch*np.ones((2048, 2048))
            
            # convertir les pixel au dessus de 255 à 255
            channel = np.uint8(np.clip(channel, 0, 255))
            #channel = channel.astype(np.uint8)
            
            #channel =  Gaussian_noise(channel, sigma_red_channel)
            #channel = Add_Salt(channel, amount_SP*0.6) 
        
        # Ajouter S & P seulement sur la channal vert 
        elif i ==1 :
            #Ajouter du bruit: parasites de photons
            
            channel = channel + Parasites_green_ch*np.ones((2048, 2048))
            channel = np.uint8(np.clip(channel, 0, 255))
            #channel = channel.astype(np.uint8)
            #channel =  Gaussian_noise(channel, sigma_green_ch)
            #channel = channel.astype(np.uint8)
            # le poucentage de salt vs pepper
            channel = Add_Salt(channel, amount_SP)
          
            
        elif i==2:
            channel = channel + (Parasites_green_ch/2)*np.ones((2048, 2048))
            channel = np.uint8(np.clip(channel, 0, 255))
        # blur each channel
        #channel_blur = gaussian_filter(channel, sigma=gaussian_Blur_sigma)
        
        noisy_image.append(channel)
        
        '''
        # enlever les valeurs négatives et les valeurs >255
        for i in range(channel.shape[0]):
            for j in range(channel.shape[1]):
                if channel[i][j]<0:
                    channel[i][j] = 0
                elif channel[i][j]>255:
                    channel[i][j] = 255
        '''
    noisy_image = np.array(noisy_image)
    noisy = noisy_image.transpose(1, 2, 0)
    
    #plt.imshow(noisy)
    return noisy







"""


from PIL import Image, ImageFilter
amount_SP = 0.1
sigma_green_channel =5
sigma_red_channel = 0.1
gaussian_Blur_sigma = 2
Parasites_green_ch = 60
Parasites_red_ch =70

image_path = './Essai/image_6.png'
image = plt.imread(image_path)
image = image[:, :, :3]
image = image*255

noisy = Add_Electronic_noise(image, amount_SP, sigma_green_channel, gaussian_Blur_sigma, Parasites_green_ch, Parasites_red_ch)
#noisy = (noisy*255).astype(np.uint8)
radius = np.random.uniform(1, 4)  
print('radius', radius)
pil_image=Image.fromarray(noisy)
pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius = radius))

pil_image.show()
pil_image.save('./Essai/image_6_.png')
"""