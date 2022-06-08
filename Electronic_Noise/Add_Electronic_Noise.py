import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random
from .Electronic_Noise_Functions import *
import time


def Add_Electronic_noise(image, glue_dir, prob_glow, prob_green_dominant, amount_SP, min_size_noise, max_size_noise, min_val_salt, max_val_salt,sigma_Gaussian_noise, 
                         gaussian_Blur_sigma, Parasites_blue_ch, Parasites_green_ch, Parasites_red_ch, Prob_change_intensity,gradient_value, Prob_Add_PSF, number_psf_max, psf_min, psf_max):
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
    Coller des morceau de taches flurescentes copiées des images réelles
    '''
    image = Add_glow(image, glue_dir, prob_glow)
    
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
    number_psf = np.random.randint(1, number_psf_max)

    for i, (channel, Parasites_ch, noise_value) in enumerate(zip(channel_list, Parasites_ch, noise_values)):
        image[:, :, i] = Add_channel_noise(channel, i, dominant_channel, amount_SP, noise_value, size_noise, sigma_Gaussian_noise, gaussian_Blur_sigma, Parasites_ch)
    
    '''
    générer une intensité de distribution gaussiene dans une zone définit
    '''
    if Adding_PSF == ['true']:
        for psf in range(number_psf):
            # a: la hauteur du picle, m: la position du centre du pic, s: sigma, indice 1 et 2 --> selon x et y
            
            ax = np.random.randint(psf_min, psf_max)
            ay = np.random.randint(psf_min, psf_max)
            mx = np.random.randint(0, row)
            my = np.random.randint(0, col)
            sx = np.random.randint(psf_min, psf_max)
            sy = np.random.randint(psf_min, psf_max)
            
            # l'indice du channal qui aura la couleur lumineuse dominante
            channel_psf = random.choices([0, 1, 2], weights=[0.35, 0.6, 0.05])
            image = Add_PSF_to_image(image, channel_psf, mx, my, ax, sx, ay, sy)
            add_near_psf = random.choices(['true', 'false'], weights=[0.7, 0.3])
            
            # ajouter d'autres tache flurescente à cote pour faire des formes différentes
            if add_near_psf ==['true']:
                for near_psf in range(3):
                    n = np.random.randint(5, 20)
                    m = np.random.randint(5, 20)
                    l = np.random.uniform(0.2, 2)
                    image = Add_PSF_to_image(image, channel_psf, mx+n, my+m, ax*l, sx, ay*l, sy)
            

  
    '''
    changer l'intensité sur certaine zone de l'image horizontalement et verticalement
    '''
    if Adding_gradients == ['true']:

        r = random.randint(-gradient_value, gradient_value)
        g = random.randint(-gradient_value, gradient_value)
        b = random.randint(-gradient_value, gradient_value)
        array_rgb = get_gradient_3d(row,col, (r, g, b), (0, 0, 0))
        image = image - array_rgb
    
        image = np.uint8(np.clip(image, 0, 255))

      
    print('Electronic noise', "--- %s seconds ---" % (time.time() - start_time))        
    return image


from PIL import Image, ImageFilter
'''
# Add glow 
glue_dir = './glow/'
prob_glow = 0.1
prob_green_dominant = 0.8
amount_SP = 0.01
min_size_noise = 2
max_size_noise = 4
min_val_salt = 80
max_val_salt = 200
sigma_Gaussian_noise =10

gaussian_Blur_sigma = 0.5
Parasites_green_ch = 60
Parasites_red_ch =80
Parasites_blue_ch = 30

image_path = './image_6_mask.png'
image = plt.imread(image_path)
image = image[:, :, :3]
image = image*255

Prob_change_intensity = 0.9
gradient_value = 30
Prob_Add_PSF = 0.9
number_psf_max = 5
# a: la hauteur du picle, m: la position du centre du pic, s: sigma, indice 1 et 2 --> selon x et y
# les plages des valeurs ont été choisi après plusieurs essais
psf_min = 5
psf_max = 20


noisy = Add_Electronic_noise(image, glue_dir, prob_glow, prob_green_dominant, amount_SP, min_size_noise, max_size_noise, min_val_salt, max_val_salt,sigma_Gaussian_noise, 
                         gaussian_Blur_sigma, Parasites_blue_ch, Parasites_green_ch, Parasites_red_ch, Prob_change_intensity,gradient_value, Prob_Add_PSF, number_psf_max, psf_min, psf_max)
radius = 1 
print('radius', radius)
pil_image=Image.fromarray(noisy)
pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius = radius))

pil_image.show()

'''