import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random
from Electronic_Noise_Functions import degraded_fibers, Add_glow, Gaussian_noise, Gaussian_noise_RGB, Add_Salt, get_gradient_3d, get_gradient_2d
import time


def Add_Electronic_noise(image, glue_dir, prob_glow, amount_SP, sigma_Gaussian_noise, gaussian_Blur_sigma, Parasites_blue_ch, Parasites_green_ch, Parasites_red_ch, Prob_change_intensity):
    start_time = time.time()
    
    assert np.amax(image)==255 and np.amin(image)==0, 'values are not in the range [0 255], normalise image values'
    row,col,ch= image.shape
    
    '''
    Dégrader les fibres et les analogues sur chaque channal en ajoutant du bruit gaussian 
    et en remplassant les valeur inférieur d'une valeurs données par 0'
    '''
    # le bruit gaussian ajouté au début pour dégrader (diminuer la qualité des fibres)
    sigma = np.random.randint(10, 50)
    
    output = degraded_fibers(image, sigma)  
    
    '''
    Coller des morceau de taches flurescentes copiées des images réelles
    '''
    output = Add_glow(output, glue_dir, prob_glow)
    
    
    '''
    Ajouter les différents type de bruit sur chaque channal separemment
    '''

    # séparer les chanaux
    red_channel = output[:, :, 0]
    green_channel = output[:, :, 1]
    blue_channel = output[:, :, 2]
    
    # choisir le channal avec plus de bruit
    
    dominant_channel = ['green', 'red']
    # la probabilité que le channal vert soit le channal dominant
    Prob = 0.9
    choosen_channel = random.choices(dominant_channel, weights=[Prob, 1-Prob])
    
    
    # Ajouter les différents type de bruit sur le channal rouge
    
    #Ajouter du bruit: parasites de photons
    red_channel = red_channel + Parasites_red_ch*np.ones((row,col))
    s_red = np.random.randint(2, 4)
    red_noise_value = np.random.randint(80, 150)
   
    if choosen_channel==['red']:
        red_channel = Add_Salt(red_channel, amount_SP, noise_value=red_noise_value, size=s_red)  
        red_channel = gaussian_filter(red_channel, sigma=gaussian_Blur_sigma)
        green_channel = Add_Salt(green_channel, amount_SP*0.1, noise_value=red_noise_value, size=s_red) 
        
    #red_channel = gaussian_filter(red_channel, sigma=gaussian_Blur_sigma)
    red_channel =  Gaussian_noise(red_channel, sigma_Gaussian_noise)
    red_channel = gaussian_filter(red_channel, sigma=gaussian_Blur_sigma)
    # convertir les pixel au dessus de 255 à 255
    red_channel = np.uint8(np.clip(red_channel, 0, 255))
     
        
    # Ajouter les différents type de bruit sur le channal vert 
    
    # Add salt (impulsive noise) sur le channal vert
    
    # Ajouter du bruit: parasites de photons
    green_channel = green_channel + Parasites_green_ch*np.ones((row,col))
    
    green_noise_value = np.random.randint(80, 200)
    if choosen_channel==['green']:
        s_green = np.random.randint(2, 4)
        green_channel = Add_Salt(green_channel, amount_SP, noise_value=green_noise_value, size=s_green) 
        red_channel = Add_Salt(red_channel, amount_SP*0.01, noise_value=green_noise_value, size=s_green)  
        green_channel = gaussian_filter(green_channel, sigma=gaussian_Blur_sigma)
    
    green_channel =  Gaussian_noise(green_channel, sigma_Gaussian_noise)    
    # blur each channel
    green_channel = gaussian_filter(green_channel, sigma=gaussian_Blur_sigma)
    green_channel = np.uint8(np.clip(green_channel, 0, 255))

    
    
    # Ajouter les différents type de bruit sur le channal bleu
      
    blue_channel = blue_channel + (Parasites_blue_ch)*np.ones((row,col))
    s_blue = np.random.randint(2, 4)
    blue_noise_value = np.random.randint(80, 150)
    blue_channel = Add_Salt(blue_channel, amount_SP*0.1, noise_value=blue_noise_value, size=s_blue) 
    blue_channel =  Gaussian_noise(blue_channel, sigma_Gaussian_noise)
    blue_channel = gaussian_filter(blue_channel, sigma=gaussian_Blur_sigma) 
      
    blue_channel = np.uint8(np.clip(blue_channel, 0, 255))
    
    Add_gradients = ['true', 'false']
    Adding_gradients = random.choices(Add_gradients, weights=[Prob_change_intensity, 1-Prob_change_intensity])  
        
    # concatener tous les chanaux
    noisy = np.dstack((red_channel, green_channel, blue_channel))
    
    
    # changer l'intensité sur certaine zone de l'image horizontalement et verticalement
    if Adding_gradients == ['true']:
        print('change intensity')
        position_r = random.choices(['true', 'false'], weights=[0.5, 0.5])
        position_g = random.choices(['true', 'false'], weights=[0.5, 0.5])
        position_b = random.choices(['true', 'false'], weights=[0.5, 0.5])
        r = random.randint(-30, 30)
        g = random.randint(-30, 50)
        b = random.randint(-30, 30)
        array_rgb = get_gradient_3d(row,col, (r, g, b), (0, 0, 0), (position_r[0], position_g[0], position_b[0]))
        noisy = noisy - array_rgb
    
        noisy = np.uint8(np.clip(noisy, 0, 255))
            
            
    print('Electronic noise', "--- %s seconds ---" % (time.time() - start_time))        
    return noisy


'''
from PIL import Image, ImageFilter

# Add glow 
glue_dir = './glow/'
prob_glow = 0.9

amount_SP = 0.01
sigma_Gaussian_noise =15

gaussian_Blur_sigma = 1
Parasites_green_ch = 60
Parasites_red_ch =80
Parasites_blue_ch = 30

image_path = './simulated_images/image_6_mask.png'
image = plt.imread(image_path)
image = image[:, :, :3]
image = image*255

Prob_change_intensity = 0.9
noisy = Add_Electronic_noise(image, glue_dir, prob_glow, amount_SP, sigma_Gaussian_noise, gaussian_Blur_sigma, Parasites_blue_ch, Parasites_green_ch, Parasites_red_ch, Prob_change_intensity)
#noisy = (noisy*255).astype(np.uint8)
radius = 1 
print('radius', radius)
pil_image=Image.fromarray(noisy)
#pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius = radius))

pil_image.show()
'''
