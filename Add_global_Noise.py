from Add_biological_noise import Add_biological_noise
from Add_flurescent_Noise import Add_Flurescent_noise
from Add_Electronic_Noise import Add_Electronic_noise
from utils import sorted_file, canvas2rgb_array
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance
from os import walk
from os.path import join



def Add_global_noise(Images_dir, Curves_paths, min_noisy_fibers, max_noisy_fibers, Prob_perlage, min_N_pixels_perlage, 
                     max_lenght_perlage, glue_dir, prob_glow, min_amount_salt, max_amount_salt, sigma_green_ch, sigma_red_channel, max_gaussian_Blur_sigma, 
                     Parasites_green_ch_max, Parasites_red_ch_max, min_Flurescent_noise, max_Flurescent_noise, min_noisy_points, max_noisy_points,output_dir_path):
    
    
    for (dirpath, dirnames, filenames) in walk(Images_dir):
        
        file_index = 0
        for image_file in sorted_file(filenames):
            # le nombre du bruit des petits fibres à ajouter à chaque image
            total_noisy_fibers = np.random.randint(min_noisy_fibers, max_noisy_fibers)
            noisy_points = np.random.randint(min_noisy_points, max_noisy_points)
            
            image_path = join(Images_dir, image_file)
            
            # Ajouter du bruit biologique à l'image: quelques petits fibres et analogues + perlage
            fig_1 = Add_biological_noise(image_path, Curves_paths, total_noisy_fibers, Prob_perlage, min_N_pixels_perlage, max_lenght_perlage, noisy_points)
            image_bio_noise = canvas2rgb_array(fig_1.canvas)
            
            
            # Ajouter le bruit électronique : Gaussian noise, S&P , Gaussian Blur, Add_constant 
            amount_salt = np.random.uniform(min_amount_salt, max_amount_salt)
            gaussian_Blur_sigma = np.random.uniform(0.5, max_gaussian_Blur_sigma) 
            Parasites_green_ch = np.random.randint(Parasites_green_ch_max-10, Parasites_green_ch_max)
            Parasites_red_ch = np.random.randint(Parasites_red_ch_max-10, Parasites_red_ch_max)
            noisy_image = Add_Electronic_noise(image_bio_noise,  glue_dir, prob_glow, amount_salt, sigma_green_ch, sigma_red_channel, gaussian_Blur_sigma, Parasites_green_ch, Parasites_red_ch)
            
            
            #Add flurescent noise 
            Flurescent_noise = np.random.randint(min_Flurescent_noise, max_Flurescent_noise)
            # noisy_points = np.random.randint(min_noisy_points, max_noisy_points)
            flurescent_image = Add_Flurescent_noise(noisy_image, Flurescent_noise, noisy_points)
            flurescent_image = canvas2rgb_array(flurescent_image.canvas)
            
            # blur image
            image_final = np.uint8(np.clip(flurescent_image, 0, 255))
            radius = np.random.uniform(0.6, 1.5)  
            pil_image=Image.fromarray(image_final)
            pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius = radius))
            
            pil_image.save(output_dir_path+'image_'+str(file_index)+'.png')
            
            print('image_', file_index, 'amount_salt', amount_salt, 'gaussian_Blur_sigma', gaussian_Blur_sigma, 'Parasites_green_ch', Parasites_green_ch, 'Parasites_red_ch', Parasites_red_ch, 'radius', radius)
            file_index +=1
        
            
            
    
            
          
'''
paramètres de chaque fonction
'''
#Add_biologic_noise
min_noisy_fibers = 20
max_noisy_fibers = 80

Prob_perlage = np.random.uniform(0.7, 1)
# la longueur min de pixel pour avoir un perlage 
min_N_pixels_perlage = 80
max_lenght_perlage = 5
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
prob_glow = 0.9

#Add_Electrnic_noise
min_amount_salt = 0.03
max_amount_salt = 0.1
sigma_green_ch =10
sigma_red_channel = 10
max_gaussian_Blur_sigma = 1
Parasites_green_ch = 70
Parasites_red_ch = 100


#Add_flurescent_noise
min_Flurescent_noise = 50
max_Flurescent_noise = 100
min_noisy_points = 500
max_noisy_points = 2000

#output_dir_path
output_dir_path = './output_images/'


Images_dir = './images/'     


Add_global_noise(Images_dir, Curves_paths, min_noisy_fibers, max_noisy_fibers, Prob_perlage, min_N_pixels_perlage, 
                     max_lenght_perlage, glue_dir, prob_glow, min_amount_salt, max_amount_salt, sigma_green_ch, sigma_red_channel, max_gaussian_Blur_sigma, 
                     Parasites_green_ch, Parasites_red_ch, min_Flurescent_noise, max_Flurescent_noise, min_noisy_points, max_noisy_points,output_dir_path)