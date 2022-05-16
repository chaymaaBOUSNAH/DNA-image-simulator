from Add_biological_noise import Add_biological_noise
from Add_flurescent_Noise import Add_Flurescent_noise
from Add_Electronic_Noise import Add_Electronic_noise
from utils import sorted_file, canvas2rgb_array

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageFilter, ImageEnhance
from os import walk
from os.path import join



def Add_global_noise(Images_dir, Curves_paths, min_noisy_fibers, max_noisy_fibers, Prob_perlage, min_N_pixels_perlage, max_lenght_perlage, min_amount_salt, max_amount_salt, sigma_green_ch, sigma_red_channel, gaussian_Blur_sigma, Parasites_green_ch, Parasites_red_ch, Flurescent_noise, pepper, output_dir_path):
    
    
    for (dirpath, dirnames, filenames) in walk(Images_dir):
        
        file_index = 0
        for image_file in sorted_file(filenames):
            # le nombre du bruit des petits fibres à ajouter à chaque image
            total_noisy_fibers = np.random.randint(min_noisy_fibers, max_noisy_fibers)
            
            image_path = join(Images_dir, image_file)
            
            # Ajouter du bruit biologique à l'image: quelques petits fibres et analogues + perlage
            fig_1 = Add_biological_noise(image_path, Curves_paths, total_noisy_fibers, Prob_perlage, min_N_pixels_perlage, max_lenght_perlage)
            image_bio_noise = canvas2rgb_array(fig_1.canvas)
        
            
            # Ajouter le bruit électronique : Gaussian noise, S&P , Gaussian Blur, Add_constant 
            amount_salt = np.random.uniform(min_amount_salt, max_amount_salt)
            noisy_image = Add_Electronic_noise(image_bio_noise, amount_salt, sigma_green_ch, sigma_red_channel, gaussian_Blur_sigma, Parasites_green_ch, Parasites_red_ch)
            
            image_final = Add_Flurescent_noise(noisy_image, Flurescent_noise, pepper)
            image_final = canvas2rgb_array(image_final.canvas)
            # blur image
            image_final = np.uint8(np.clip(image_final, 0, 255))
            radius = np.random.uniform(1, 3)  
            pil_image=Image.fromarray(image_final)
            pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius = radius))
            
            pil_image.save(output_dir_path+'image_'+str(file_index)+'.png')
            file_index +=1
        
            
            
    
            
          
'''
paramètres de chaque fonction
'''
#Add_biologic_noise
min_noisy_fibers = 20
max_noisy_fibers = 50
noisy_dust = np.random.randint(50, 200)
pepper = np.random.randint(6000, 12000)
perlage = np.random.randint(500, 2000)
prob = np.random.uniform(0.8, 1)
# la longueur min de pixel pour avoir un perlage 
min_N_pixels_perlage = 100
max_lenght_perlage = 80
#lire les fichier csv des coordonnées des courbes
files_path1 = './Curves_data/L_U1/'
files_path2 = './Curves_data/L_U2/'
files_path3 = './Curves_data/L_C11/'    
files_path4 = './Curves_data/L_C12/' 
files_path5 = './Curves_data/L_C21/' 
files_path6 = './Curves_data/L_C22/' 
files_path7 = './fibres_data/' 

Curves_paths = [files_path1, files_path2, files_path3, files_path4,
                files_path5, files_path6, files_path7]


#Add_Electrnic_noise
min_amount_salt = 0.01
max_amount_salt = 0.3
sigma_green_ch =5
sigma_red_channel = 0.1
gaussian_Blur_sigma = 2
Parasites_green_ch = 60
Parasites_red_ch = 80




#Add_flurescent_noise
Flurescent_noise = np.random.randint(100, 500)
pepper = np.random.randint(6000, 12000)

#output_dir_path
output_dir_path = './output_images/'


Images_dir = './images/'     


Add_global_noise(Images_dir, Curves_paths, min_noisy_fibers, max_noisy_fibers, prob, min_N_pixels_perlage, max_lenght_perlage, min_amount_salt, max_amount_salt, sigma_green_ch, sigma_red_channel, gaussian_Blur_sigma, Parasites_green_ch, Parasites_red_ch, Flurescent_noise, pepper, output_dir_path)

       