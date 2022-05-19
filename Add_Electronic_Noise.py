import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random
from Add_gaussian_2D_image import Gaussian_noise, Gaussian_noise_RGB, Add_Salt

def Add_Electronic_noise(image, amount_SP, sigma_green_ch, sigma_red_channel, gaussian_Blur_sigma, Parasites_green_ch, Parasites_red_ch):
    Prob = 0.8
    row,col,ch= image.shape
    
    output = (image).astype(np.uint8)
    
    blue_channel = output[:, :, 2]
    red_channel = output[:, :, 0]
    green_channel = output[:, :, 1]

    dominant_channel = ['green', 'red']
    choosen_channel = random.choices(dominant_channel, weights=[Prob, 1-Prob])
    
    # Ajouter les différents type de bruit sur le channal rouge
    
    #Ajouter du bruit: parasites de photons
    red_channel = red_channel + Parasites_red_ch*np.ones((row,col))
    red_channel =  Gaussian_noise(red_channel, sigma_red_channel)
    # convertir les pixel au dessus de 255 à 255
    red_channel = np.uint8(np.clip(red_channel, 0, 255))
    if choosen_channel==['red']:
        red_channel = Add_Salt(red_channel, amount_SP) 
    # blur each channel
    red_channel = gaussian_filter(red_channel, sigma=gaussian_Blur_sigma)
     
        
    # Ajouter les différents type de bruit sur le channal vert 
            
    # Ajouter du bruit: parasites de photons
    green_channel = green_channel + Parasites_green_ch*np.ones((row,col))
    green_channel =  Gaussian_noise(green_channel, sigma_green_ch)
    green_channel = np.uint8(np.clip(green_channel, 0, 255))
    #channel = channel.astype(np.uint8)
    if choosen_channel==['green']:
        green_channel = Add_Salt(green_channel, amount_SP)
    else: 
       green_channel = Add_Salt(green_channel, amount_SP*0.5)
    # blur each channel
    green_channel = gaussian_filter(green_channel, sigma=gaussian_Blur_sigma)
            
    
    # Ajouter les différents type de bruit sur le channal bleu
        
    blue_channel = blue_channel + (Parasites_green_ch/2)*np.ones((row,col))
    blue_channel =  Gaussian_noise(blue_channel, sigma_red_channel)
    blue_channel = np.uint8(np.clip(blue_channel, 0, 255))
    # blur each channel
    blue_channel = gaussian_filter(blue_channel, sigma=gaussian_Blur_sigma)
        
    # concatener tous les chanaux
    noisy = np.dstack((red_channel, green_channel, blue_channel))
    
    return noisy



'''
from PIL import Image, ImageFilter
amount_SP = 0.1
sigma_green_channel =1
sigma_red_channel = 0.1
gaussian_Blur_sigma = 0.6
Parasites_green_ch = 40
Parasites_red_ch =50

image_path = './Essai/image_0.png'
image = plt.imread(image_path)
image = image[:, :, :3]
image = image*255
Prob = 0.5

noisy = Add_Electronic_noise(image, amount_SP, sigma_green_channel,sigma_red_channel, gaussian_Blur_sigma, Parasites_green_ch, Parasites_red_ch)
#noisy = (noisy*255).astype(np.uint8)
radius = np.random.uniform(1, 2)  
print('radius', radius)
pil_image=Image.fromarray(noisy)
pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius = radius))

pil_image.show()
pil_image.save('./Essai/image_0_1.png')
'''