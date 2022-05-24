import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random
from Electronic_Noise_Functions import degraded_fibers, Add_glow, Gaussian_noise, Gaussian_noise_RGB, Add_Salt


def Add_Electronic_noise(image, glue_dir, prob_glow, amount_SP, sigma_green_ch, sigma_red_channel, Parasites_blue_ch, gaussian_Blur_sigma, Parasites_green_ch, Parasites_red_ch):
    # la probabilité que le channal vert soit le channal dominant
    Prob = 1
    # le bruit gaussian ajouté au début pour dégrader (diminuer la qualité des fibres)
    sigma = np.random.randint(10, 30)
    row,col,ch= image.shape
    
    
    assert np.amax(image)==255 and np.amin(image)==0, 'values are not in the range [0 255], normalise image values'
    
    
    '''
    Dégrader les fibres et les analogues sur chaque channal en ajoutant du bruit gaussian 
    et en remplassant les valeur inférieur d'une valeurs données '
    '''
    
    output = degraded_fibers(image, sigma)  
    
    '''
    Coller des morceau de taches flurescentes copiées des images réelles
    '''
    output = Add_glow(output, glue_dir, prob_glow)

    dominant_channel = ['green', 'red']
    choosen_channel = random.choices(dominant_channel, weights=[Prob, 1-Prob])
    
    red_channel = output[:, :, 0]
    green_channel = output[:, :, 1]
    blue_channel = output[:, :, 2]
    
    '''Ajouter les différents type de bruit sur chaque channal separemment'''
    
    # Ajouter les différents type de bruit sur le channal rouge
    
    

    #Ajouter du bruit: parasites de photons
    red_channel = red_channel + Parasites_red_ch*np.ones((row,col))
    s_red = np.random.randint(2, 6)
    red_noise_value = np.random.randint(80, 200)
   
    if choosen_channel==['red']:
        red_channel = Add_Salt(red_channel, amount_SP, noise_value=red_noise_value, size=s_red)  
        green_channel = Add_Salt(green_channel, amount_SP*0.1, noise_value=red_noise_value, size=s_red) 
        
    #red_channel = gaussian_filter(red_channel, sigma=gaussian_Blur_sigma)
    red_channel =  Gaussian_noise(red_channel, sigma_red_channel)
    red_channel = gaussian_filter(red_channel, sigma=gaussian_Blur_sigma)
    # convertir les pixel au dessus de 255 à 255
    red_channel = np.uint8(np.clip(red_channel, 0, 255))
     
        
    # Ajouter les différents type de bruit sur le channal vert 
    
    # Add salt (impulsive noise) sur le channal vert
    
    # Ajouter du bruit: parasites de photons
    green_channel = green_channel + Parasites_green_ch*np.ones((row,col))
    
    green_noise_value = np.random.randint(80, 200)
    if choosen_channel==['green']:
        s_green = np.random.randint(2, 6)
        green_channel = Add_Salt(green_channel, amount_SP, noise_value=green_noise_value, size=s_green) 
        red_channel = Add_Salt(red_channel, amount_SP*0.01, noise_value=green_noise_value, size=s_green)  
     #green_channel = gaussian_filter(green_channel, sigma=gaussian_Blur_sigma)
    
    green_channel =  Gaussian_noise(green_channel, sigma_green_ch)
    green_channel = gaussian_filter(green_channel, sigma=gaussian_Blur_sigma)
    
    # blur each channel
    green_channel = gaussian_filter(green_channel, sigma=gaussian_Blur_sigma)
    green_channel = np.uint8(np.clip(green_channel, 0, 255))
    #channel = channel.astype(np.uint8)
    
    # blur each channel
    #green_channel = gaussian_filter(green_channel, sigma=gaussian_Blur_sigma)
            
    
    # Ajouter les différents type de bruit sur le channal bleu
      
    blue_channel = blue_channel + (Parasites_blue_ch)*np.ones((row,col))
    s_blue = np.random.randint(2, 6)
    blue_noise_value = np.random.randint(80, 200)
    blue_channel = Add_Salt(blue_channel, amount_SP*0.1, noise_value=blue_noise_value, size=s_blue) 
    #blue_channel = gaussian_filter(blue_channel, sigma=gaussian_Blur_sigma) 
    blue_channel =  Gaussian_noise(blue_channel, sigma_red_channel)
    blue_channel = gaussian_filter(blue_channel, sigma=gaussian_Blur_sigma) 
    
    
    blue_channel = np.uint8(np.clip(blue_channel, 0, 255))
    
    # blur each channel
    
        
    # concatener tous les chanaux
    noisy = np.dstack((red_channel, green_channel, blue_channel))
    
    return noisy



from PIL import Image, ImageFilter
'''
# Add glow 
glue_dir = './glow/'
prob_glow = 0.9

amount_SP = 0.001
sigma_green_channel =15
sigma_red_channel =10
gaussian_Blur_sigma = 1
Parasites_green_ch = 60
Parasites_red_ch =80
Parasites_blue_ch = 30

image_path = './Essai/image_6_.png'
image = plt.imread(image_path)
image = image[:, :, :3]
image = image*255


noisy = Add_Electronic_noise(image, glue_dir, prob_glow, amount_SP, sigma_green_channel,sigma_red_channel,Parasites_blue_ch, gaussian_Blur_sigma, Parasites_green_ch, Parasites_red_ch)
#noisy = (noisy*255).astype(np.uint8)
radius = np.random.uniform(1, 2)  
print('radius', radius)
pil_image=Image.fromarray(noisy)
#pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius = radius))

pil_image.show()
pil_image.save('./Essai/image_0_1.png')
'''