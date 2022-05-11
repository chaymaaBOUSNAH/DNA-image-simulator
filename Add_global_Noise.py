from Add_biological_noise import Add_biological_noise
from Add_flurescent_Noise import Add_Flurescent_noise
from utils import sorted_file, canvas2rgb_array

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageFilter, ImageEnhance
from os import walk
from os.path import join

'''
paramètres de chaque fonction
'''

total_noisy_fibers = np.random.randint(20, 50)
noisy_dust = np.random.randint(50, 200)
pepper = np.random.randint(6000, 12000)
perlage = np.random.randint(500, 2000)
prob = np.random.uniform(0.5, 0.9)
max_num_perlage = 5
max_discontuinity = 50
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

#Add_flurescent_noise
Flurescent_noise = np.random.randint(50, 200)
pepper = np.random.randint(6000, 12000)




Images_dir = './create_images/noisy_curves/'

def Add_global_noise(Images_dir, Curves_paths, Prob_perlage= 0.8):
    
    
    for (dirpath, dirnames, filenames) in walk(Images_dir):
        
        file_index = 0
        for image_file in sorted_file(filenames):
            
            image_path = join(Images_dir, image_file)
            
            # Ajouter du bruit biologique à l'image: quelques petits fibres et analogues et le perlage
            fig_1 = Add_biological_noise(image_path, Curves_paths, total_noisy_fibers, prob, max_num_perlage, max_discontuinity)
            image_bio_noise = canvas2rgb_array(fig_1.canvas)
            
            print(image_bio_noise.shape)
            
            # Ajouter une constante (parasites de lumière) sur chaque channal
            
            
            
            
            