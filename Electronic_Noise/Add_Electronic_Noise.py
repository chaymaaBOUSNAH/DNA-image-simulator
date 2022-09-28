import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random
from .Electronic_Noise_Functions import *
import time


def Add_Electronic_noise(image, prob_green_dominant, amount_SP, min_size_noise, max_size_noise, min_val_salt, max_val_salt,sigma_Gaussian_noise, 
                         gaussian_Blur_sigma, Parasites_blue_ch, Parasites_green_ch, Parasites_red_ch, Prob_change_intensity,gradient_value, Prob_Add_PSF, number_psf_max, psf_min, psf_max, scanner_img):
    start_time = time.time()
    
    assert np.amax(image)==255 and np.amin(image)==0, 'values are not in the range [0 255], normalise image values'
    assert len(image.shape) == 3 and image.shape[-1] == 3, 'Verify image shape'
    
    row,col,ch= image.shape
    
    '''
    Dégrader les fibres et les analogues sur chaque channal en ajoutant du bruit gaussian 
    et en remplassant les valeur inférieur d'une valeurs données par 0 : appliquer mask <255 ==0'
    '''
    # Bruit gaussian ajouté au début pour dégrader (diminuer la qualité des fibres)
    sigma = np.random.randint(10, 50)  
    image = degraded_fibers(image, sigma)  
    
    
    '''
    Ajouter les différents type de bruit sur chaque channal separemment: Salt, Gaussian noise, Blur
    '''
    # choisir le channal dominant : avec plus de bruit  
    # la probabilité que le channal vert soit le channal dominant
    Prob = prob_green_dominant
    dominant_channel = random.choices(['green', 'red'], weights=[Prob, 1-Prob])
    # la taille du sel à ajouter à chque channal 
    size_noise = np.random.randint(min_size_noise, max_size_noise)
    # la valeurs des pixel du bruit de sel pour chaque channal
    noise_value_R = np.random.randint(min_val_salt, max_val_salt-50)
    noise_value_G = np.random.randint(min_val_salt, max_val_salt)
    noise_value_B = np.random.randint(min_val_salt, max_val_salt)
    noise_values = (noise_value_R, noise_value_G, noise_value_B)
    channel_list = split_channels(image)
    # parasites de lumière sur chaque channal
    Parasites_ch = (Parasites_red_ch, Parasites_green_ch, Parasites_blue_ch)
    # changer l'intensité sur certaine zone de chaque channal horizontalement et verticalement
    Adding_gradients = random.choices(['true', 'false'], weights=[Prob_change_intensity, 1-Prob_change_intensity])

    # paramètres pour générer une intensité de distribution gaussiene dans une zone définit 
    Adding_PSF = random.choices(['true', 'false'], weights=[Prob_Add_PSF, 1-Prob_Add_PSF])
    number_psf = np.random.randint(2, number_psf_max)

    for i, (channel, Parasites_ch, noise_value) in enumerate(zip(channel_list, Parasites_ch, noise_values)):
        image[:, :, i] = Add_channel_noise(channel, i, dominant_channel, amount_SP, noise_value, size_noise, sigma_Gaussian_noise, gaussian_Blur_sigma, Parasites_ch)
    
    '''
    générer une intensité de distribution gaussiene dans une zone définit
    '''
    if Adding_PSF == ['true']:
        for psf in range(number_psf):
            # a: la hauteur du picle, m: la position du centre du pic, s: sigma, indice 1 et 2 --> selon x et y
            
            ax = np.random.uniform(psf_min, psf_max)
            ay = np.random.uniform(psf_min, psf_max)
            mx = np.random.uniform(0, row)
            my = np.random.uniform(0, col)
            sx = np.random.uniform(psf_min, psf_max)
            sy = np.random.uniform(psf_min, psf_max)
            
            # l'indice du channal qui aura la couleur lumineuse dominante
            channel_psf = random.choices([0, 1, 2], weights=[0.35, 0.6, 0.05])
            image = Add_PSF_to_image(image, channel_psf, mx, my, ax, sx, ay, sy)

    '''
    changer l'intensité sur certaine zone de l'image horizontalement et verticalement
    '''
    if Adding_gradients == ['true']:

        r = random.uniform(-gradient_value, gradient_value)
        g = random.uniform(-gradient_value, gradient_value)
        b = random.uniform(-gradient_value, gradient_value)
        array_rgb = get_gradient_3d(row,col, (r, g, b), (0, 0, 0))
        image = image - array_rgb
        image = np.uint8(np.clip(image, 0, 255))
    
        
    
    if scanner_img:
        
        global_sigma = np.random.randint(60, 90)
        image = Gaussian_noise_RGB(image, global_sigma)

      
    print('Electronic noise', "--- %s seconds ---" % (time.time() - start_time))        
    return image


from PIL import Image, ImageFilter
