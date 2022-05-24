from Add_biological_noise import Add_biological_noise
from Add_flurescent_Noise import Add_Flurescent_noise
from Add_Electronic_Noise import Add_Electronic_noise
from Electronic_Noise_Functions import Gaussian_noise_RGB
from utils import sorted_file, canvas2rgb_array
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance
from os import walk
from os.path import join



def Add_global_noise(Images_dir, Curves_paths, min_noisy_fibers, max_noisy_fibers, Prob_perlage, min_N_pixels_perlage, 
                     max_lenght_perlage, glue_dir, prob_glow, min_amount_salt, max_amount_salt, sigma_green_ch_max, sigma_red_channel_max, max_gaussian_Blur_sigma, 
                     Parasites_green_ch_max, Parasites_red_ch_max, Parasites_blue_ch_max, min_Flurescent_noise, max_Flurescent_noise, min_noisy_points, max_noisy_points,output_dir_path):
    
    
    for (dirpath, dirnames, filenames) in walk(Images_dir):
        
        file_index = 0
        for image_file in sorted_file(filenames):
        
            image_path = join(Images_dir, image_file)
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            
            assert np.amax(image)==255 and np.amin(image)==0, f'{image_file} values are not in the range [0 255], normalise image values'
            
            # le nombre du bruit des petits fibres à ajouter à chaque image
            total_noisy_fibers = np.random.randint(min_noisy_fibers, max_noisy_fibers)
            noisy_points = np.random.randint(min_noisy_points, max_noisy_points)
            
            # Ajouter du bruit biologique à l'image: quelques petits fibres et analogues + perlage
            fig_1 = Add_biological_noise(image, image_file, Curves_paths, total_noisy_fibers, Prob_perlage, min_N_pixels_perlage, max_lenght_perlage, noisy_points)
            image_bio_noise = canvas2rgb_array(fig_1.canvas)
            
            
            # Ajouter le bruit électronique : Gaussian noise, S&P , Gaussian Blur, Add_constant 
            amount_salt = np.random.randint(min_amount_salt, max_amount_salt)
            gaussian_Blur_sigma = np.random.uniform(0.7, max_gaussian_Blur_sigma) 
            Parasites_green_ch = np.random.randint(Parasites_green_ch_max-20, Parasites_green_ch_max)
            Parasites_red_ch = np.random.randint(Parasites_red_ch_max-20, Parasites_red_ch_max)
            Parasites_blue_ch = np.random.randint(Parasites_blue_ch_max-20, Parasites_blue_ch_max)
            sigma_red_channel = np.random.randint(7, sigma_red_channel_max)
            sigma_green_ch = np.random.randint(7, sigma_green_ch_max)
            noisy_image = Add_Electronic_noise(image_bio_noise,  glue_dir, prob_glow, amount_salt, sigma_green_ch, sigma_red_channel, Parasites_blue_ch, gaussian_Blur_sigma, Parasites_green_ch, Parasites_red_ch)
            
            
            #Add flurescent noise 
            Flurescent_noise = np.random.randint(min_Flurescent_noise, max_Flurescent_noise)
            # noisy_points = np.random.randint(min_noisy_points, max_noisy_points)
            flurescent_image = Add_Flurescent_noise(noisy_image, Flurescent_noise, noisy_points)
            flurescent_image = canvas2rgb_array(flurescent_image.canvas)
            
            img = Gaussian_noise_RGB(flurescent_image, 10)
            # Convertir toutes les valeurs dans un intervalle de 0 255
            image_final = np.uint8(np.clip(img, 0, 255))
            
            Parasites = np.random.randint(5, 20)
            #image_final = image_final+ Parasites
            # convertir en image Pil
            pil_image=Image.fromarray(image_final)
            # Augmenter le contrast de l'image
            #enhancer = ImageEnhance.Contrast(pil_image)
            #factor = np.random.uniform(1, 2) #increase contrast
            #pil_image = enhancer.enhance(factor)
            
            # blur image
            
            radius = np.random.uniform(0.5, 1)  
            pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius = radius))
            pil_image.save(output_dir_path+'image_'+str(file_index)+'.png')
            
            print('image_', file_index, 'amount_salt', amount_salt, 'gaussian_Blur_sigma', gaussian_Blur_sigma, 'sigma_green_ch', sigma_green_ch, 'sigma_red_channel', sigma_red_channel,'Parasites_green_ch', Parasites_green_ch, 'Parasites_red_ch', Parasites_red_ch, 'Parasites_blue_ch', Parasites_blue_ch, 'radius', radius)
            file_index +=1
        
            
            
    
            
          
'''
paramètres de chaque fonction
'''
#Add_biologic_noise
min_noisy_fibers = 20
max_noisy_fibers = 80

Prob_perlage = np.random.uniform(0.7, 1)
# la longueur min de pixel pour avoir un perlage 
min_N_pixels_perlage = 50
max_lenght_perlage = 3
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

# Add glow 
glue_dir = './glow/'
prob_glow = 0.5

#Add_Electrnic_noise
min_amount_salt = 20000 
max_amount_salt = 50000
sigma_green_ch = 25
sigma_red_channel = 15
max_gaussian_Blur_sigma = 2
Parasites_green_ch = 70
Parasites_red_ch = 100
Parasites_blue_ch = 50


#Add_flurescent_noise
min_Flurescent_noise = 100
max_Flurescent_noise = 200
min_noisy_points = 50
max_noisy_points = 100

#output_dir_path
output_dir_path = './output_images/'


Images_dir = './images/'     


Add_global_noise(Images_dir, Curves_paths, min_noisy_fibers, max_noisy_fibers, Prob_perlage, min_N_pixels_perlage, 
                     max_lenght_perlage, glue_dir, prob_glow, min_amount_salt, max_amount_salt, sigma_green_ch, sigma_red_channel, max_gaussian_Blur_sigma, 
                     Parasites_green_ch, Parasites_red_ch,Parasites_blue_ch, min_Flurescent_noise, max_Flurescent_noise, min_noisy_points, max_noisy_points,output_dir_path)