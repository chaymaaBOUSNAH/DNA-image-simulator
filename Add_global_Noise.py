from Add_biological_Noise import Add_biological_noise
from Add_flurescent_Noise import Add_Flurescent_noise
from Add_Electronic_Noise import Add_Electronic_noise
from Electronic_Noise_Functions import get_gradient_3d
from utils import sorted_file, canvas2rgb_array
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance
import os
from os import walk
from os.path import join
import random
import matplotlib as mpl
mpl.rc('figure', max_open_warning = 0)


def Add_global_noise(Images_dir, csv_path, min_noisy_fibers, max_noisy_fibers, Prob_perlage, min_N_pixels_perlage, 
                     max_lenght_perlage, glue_dir, prob_glow, min_amount_salt, max_amount_salt, sigma_green_ch, sigma_red_channel, max_gaussian_Blur_sigma, 
                     Parasites_green_ch_max, Parasites_blue_ch_max, Parasites_red_ch_max, min_Flurescent_noise, max_Flurescent_noise, min_noisy_points, max_noisy_points,Prob_change_intensity, output_dir_path):
    
    if os.path.isdir(output_dir_path):
        print('directory exist !')
    else:
        os.makedirs(output_dir_path)
        print('directory created !')
        
    for (dirpath, dirnames, filenames) in walk(Images_dir):
        
        file_index = 0
        for image_file in sorted_file(filenames):
            
            image_path = join(Images_dir, image_file)
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            
            ht, wd = image.shape[:2]
            assert np.amax(image)==255 and np.amin(image)==0, f'{image_file} values are not in the range [0 255], normalise image values'
            
            # le nombre du bruit des petits fibres à ajouter à chaque image
            total_noisy_fibers = np.random.randint(min_noisy_fibers, max_noisy_fibers)
            noisy_points = np.random.randint(min_noisy_points, max_noisy_points)
            
            # Ajouter du bruit biologique à l'image: quelques petits fibres et analogues + perlage
            fig_1 = Add_biological_noise(image, image_file, csv_path, total_noisy_fibers, Prob_perlage, min_N_pixels_perlage, max_lenght_perlage, noisy_points)
            image_bio_noise = canvas2rgb_array(fig_1.canvas)
            
            
            # Ajouter le bruit électronique : Gaussian noise, S&P , Gaussian Blur, Add_constant 
            amount_salt = np.random.uniform(min_amount_salt, max_amount_salt)
            gaussian_Blur_sigma = np.random.uniform(1, max_gaussian_Blur_sigma) 
            Parasites_green_ch = np.random.randint(Parasites_green_ch_max-20, Parasites_green_ch_max)
            Parasites_red_ch = np.random.randint(Parasites_red_ch_max-20, Parasites_red_ch_max)
            Parasites_blue_ch = np.random.randint(Parasites_blue_ch_max-20, Parasites_blue_ch_max)
            noisy_image = Add_Electronic_noise(image_bio_noise,  glue_dir, prob_glow, amount_salt, sigma_green_ch, sigma_red_channel, Parasites_blue_ch, gaussian_Blur_sigma, Parasites_green_ch, Parasites_red_ch)
            
            Add_gradients = ['true', 'false']
            change_intensity = random.choices(Add_gradients, weights=[Prob_change_intensity, 1-Prob_perlage])
            
            if change_intensity == ['true']:
                print('change intensity')
                position = random.choices(['true', 'false'], weights=[0.5, 0.5])
                position_g = random.choices(['true', 'false'], weights=[0.5, 0.5])
                
                r = random.randint(-20, 20)
                g = random.randint(-50, 50)
                b = random.randint(-20, 20)
                array_rgb = get_gradient_3d(ht, wd, (r, g, b), (0, 0, 0), (position[0], position_g[0], position[0]))
                array_rgb = np.uint8(np.clip(array_rgb, 0, 255))
    
                noisy_image = noisy_image - array_rgb
    
                noisy_image = np.uint8(np.clip(noisy_image, 0, 255))
            
            #Add flurescent noise 
            Flurescent_noise = np.random.randint(min_Flurescent_noise, max_Flurescent_noise)
            # noisy_points = np.random.randint(min_noisy_points, max_noisy_points)
            flurescent_image = Add_Flurescent_noise(noisy_image, Flurescent_noise, noisy_points)
            flurescent_image = canvas2rgb_array(flurescent_image.canvas)
            
            # blur image
            image_final = np.uint8(np.clip(flurescent_image, 0, 255))
            
            radius = np.random.uniform(0.2, 0.8)
            pil_image = Image.fromarray(image_final)
            # Augmenter le contrast de l'image
            #enhancer = ImageEnhance.Contrast(pil_image)
            #factor = np.random.uniform(1, 1.5) #increase contrast
            #pil_image = enhancer.enhance(factor)
            
            pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius = radius))
            
            
            pil_image.save(output_dir_path+'image_'+str(file_index)+'.png')
            
            print('image_', file_index, 'amount_salt', amount_salt, 'gaussian_Blur_sigma', gaussian_Blur_sigma, 'Parasites_green_ch', Parasites_green_ch, 'Parasites_red_ch', Parasites_red_ch, 'Parasites_blue_ch', Parasites_blue_ch, 'radius', radius)
            file_index +=1
        
            
            
    
            
          
'''
paramètres de chaque fonction
'''
#Add_biologic_noise
min_noisy_fibers = 40
max_noisy_fibers = 100

Prob_perlage = np.random.uniform(0.7, 1)
# la longueur min de pixel pour avoir un perlage 
min_N_pixels_perlage = 50
max_lenght_perlage = 5
#lire les fichier csv des coordonnées des courbes

csv_path = './fibers_coords' 


# Add glow 
glue_dir = './glow/'
prob_glow = 0.9

#Add_Electrnic_noise
min_amount_salt = 0.0001
max_amount_salt = 0.05
sigma_green_ch =15
sigma_red_channel = 10
max_gaussian_Blur_sigma = 2
Parasites_green_ch = 70
Parasites_red_ch = 100
Parasites_blue_ch = 50

#Add_flurescent_noise
min_Flurescent_noise = 30
max_Flurescent_noise = 80
min_noisy_points = 500
max_noisy_points = 1000


Prob_change_intensity = 0.8


#output_dir_path
output_dir_path = './output_images/'


Images_dir = './simulated_images/'     


Add_global_noise(Images_dir, csv_path, min_noisy_fibers, max_noisy_fibers, Prob_perlage, min_N_pixels_perlage, 
                     max_lenght_perlage, glue_dir, prob_glow, min_amount_salt, max_amount_salt, sigma_green_ch, sigma_red_channel, max_gaussian_Blur_sigma, 
                     Parasites_green_ch, Parasites_blue_ch, Parasites_red_ch, min_Flurescent_noise, max_Flurescent_noise, min_noisy_points, max_noisy_points,Prob_change_intensity, output_dir_path)