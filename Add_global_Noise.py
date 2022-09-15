from Biologic_Noise.Add_biological_noise import Add_biological_noise
from Electronic_Noise.Add_Electronic_Noise import Add_Electronic_noise
from util import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance
import os
import json
from os import walk
from os.path import join
import random
import time
import matplotlib as mpl

mpl.rc('figure', max_open_warning=0)


def Add_global_noise(Images_dir, csv_path, min_noisy_fibers, max_noisy_fibers, min_Prob_perlage, max_Prob_perlage,
                     min_N_pixels_perlage,
                     max_lenght_perlage, prob_green_dominant, min_amount_salt, max_amount_salt, min_size_noise,
                     max_size_noise, min_val_salt, max_val_salt, min_sigma_Gaussian_noise, max_sigma_Gaussian_noise,
                     max_gaussian_Blur_sigma,
                     Parasites_green_ch_max, Parasites_blue_ch_max, Parasites_red_ch_max, Prob_change_intensity,
                     gradient_value, Prob_Add_PSF, number_psf_max, psf_min, psf_max, min_noisy_points, max_noisy_points,
                     output_dir_path, scanner_img=True):


    create_directory(output_dir_path)

    for (dirpath, dirnames, filenames) in walk(Images_dir):

        file_index = 0

        for image_file in sorted_file(filenames):
            indx = image_file.split('_')
            start_time = time.time()

            image_path = join(Images_dir, image_file)
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)

            assert np.amax(image) == 255 and np.amin(
                image) == 0, f'{image_file} values are not in the range [0 255], normalise image values'

            # le nombre du bruit des petits fibres à ajouter à chaque image
            total_noisy_fibers = np.random.randint(min_noisy_fibers, max_noisy_fibers)
            noisy_points = np.random.randint(min_noisy_points, max_noisy_points)
            # Ajouter du bruit biologique à l'image: quelques petits fibres et analogues + perlage
            image_bio_noise = Add_biological_noise(image, image_file, csv_path, total_noisy_fibers, min_Prob_perlage,
                                                   max_Prob_perlage, min_N_pixels_perlage, max_lenght_perlage,
                                                   noisy_points)

            # Ajouter le bruit électronique : Gaussian noise, S&P , Gaussian Blur, Add_constant
            amount_salt = np.random.uniform(min_amount_salt, max_amount_salt)
            gaussian_Blur_sigma = np.random.uniform(0.5, max_gaussian_Blur_sigma)
            Parasites_green_ch = np.random.randint(Parasites_green_ch_max - 20, Parasites_green_ch_max)
            Parasites_red_ch = np.random.randint(Parasites_red_ch_max - 40, Parasites_red_ch_max)
            Parasites_blue_ch = np.random.randint(Parasites_blue_ch_max - 40, Parasites_blue_ch_max)
            sigma_Gaussian_noise = np.random.randint(min_sigma_Gaussian_noise, max_sigma_Gaussian_noise)

            noisy_image = Add_Electronic_noise(image_bio_noise, prob_green_dominant, amount_salt, min_size_noise,
                                               max_size_noise, min_val_salt, max_val_salt, sigma_Gaussian_noise,
                                               gaussian_Blur_sigma, Parasites_blue_ch, Parasites_green_ch,
                                               Parasites_red_ch, Prob_change_intensity, gradient_value, Prob_Add_PSF,
                                               number_psf_max, psf_min, psf_max, scanner_img)

            # blur image
            image_final = np.uint8(np.clip(noisy_image, 0, 255))

            radius = np.random.uniform(0.2, 1)
            pil_image = Image.fromarray(image_final)
            # Augmenter le contrast de l'image
            # enhancer = ImageEnhance.Contrast(pil_image)
            # factor = np.random.uniform(1, 1.5) #increase contrast
            # pil_image = enhancer.enhance(factor)

            #pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=radius))

            pil_image.save(output_dir_path + 'image_' + indx[1] + '.png')
            plt.close()
            file_index += 1
            print("--- %s seconds ---" % (time.time() - start_time))
            print('file_index', file_index)


"""load parameters from json file """

with open('Noise_parameters.json') as json_file:
    data = json.load(json_file)
    print('Noise parameters:', data)

"""paramètres fichier de configuration:"""

# Add_biologic_noise
Add_biologic_noise = data['Add_biologic_noise']
min_noisy_fibers = Add_biologic_noise['min_noisy_fibers']
max_noisy_fibers = Add_biologic_noise['max_noisy_fibers']
min_Prob_perlage = Add_biologic_noise['min_Prob_perlage']
max_Prob_perlage = Add_biologic_noise['max_Prob_perlage']
# la longueur min de pixel pour avoir un perlage
min_N_pixels_perlage = Add_biologic_noise['min_N_pixels_perlage']
max_lenght_perlage = Add_biologic_noise['max_lenght_perlage']
min_noisy_points = Add_biologic_noise['min_noisy_points']
max_noisy_points = Add_biologic_noise['max_noisy_points']

# Add_Electrnic_noise
Add_Electrnic_noise = data['Add_Electrnic_noise']
prob_green_dominant = Add_Electrnic_noise['prob_green_dominant']
min_amount_salt = Add_Electrnic_noise['min_amount_salt']
max_amount_salt = Add_Electrnic_noise['max_amount_salt']
min_size_noise = Add_Electrnic_noise['min_size_noise']
max_size_noise = Add_Electrnic_noise['max_size_noise']
min_val_salt = Add_Electrnic_noise['min_val_salt']
max_val_salt = Add_Electrnic_noise['max_val_salt']

min_sigma_Gaussian_noise = Add_Electrnic_noise['min_sigma_Gaussian_noise']
max_sigma_Gaussian_noise = Add_Electrnic_noise['max_sigma_Gaussian_noise']
max_gaussian_Blur_sigma = Add_Electrnic_noise['max_gaussian_Blur_sigma']
Parasites_green_ch = Add_Electrnic_noise['Parasites_green_ch']
Parasites_red_ch = Add_Electrnic_noise['Parasites_red_ch']
Parasites_blue_ch = Add_Electrnic_noise['Parasites_blue_ch']
Prob_change_intensity = Add_Electrnic_noise['Prob_change_intensity']
gradient_value = Add_Electrnic_noise['gradient_value']
number_psf_max = Add_Electrnic_noise['number_psf_max']
Prob_Add_PSF = Add_Electrnic_noise['Prob_Add_PSF']
psf_min = Add_Electrnic_noise['psf_min']
psf_max = Add_Electrnic_noise['psf_max']


# paths
paths = data['Path']
csv_path = paths['csv_path']
#output_dir_path
output_dir_path = paths['output_dir_path']
Images_dir = paths['Images_dir']




Add_global_noise(Images_dir, csv_path, min_noisy_fibers, max_noisy_fibers, min_Prob_perlage, max_Prob_perlage,
                     min_N_pixels_perlage,
                     max_lenght_perlage, prob_green_dominant, min_amount_salt, max_amount_salt, min_size_noise,
                     max_size_noise, min_val_salt, max_val_salt, min_sigma_Gaussian_noise, max_sigma_Gaussian_noise,
                     max_gaussian_Blur_sigma,
                     Parasites_green_ch, Parasites_blue_ch, Parasites_red_ch, Prob_change_intensity,
                     gradient_value, Prob_Add_PSF, number_psf_max, psf_min, psf_max, min_noisy_points, max_noisy_points,
                     output_dir_path)
