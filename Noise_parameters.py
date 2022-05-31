import json

Noise_parameters = {

    
    "Add_biologic_noise": {
        "min_noisy_fibers": 40,
        "max_noisy_fibers": 100,
        "min_Prob_perlage": 0.5,
        "max_Prob_perlage": 1,
        "min_N_pixels_perlage": 50,
        "max_lenght_perlage": 5
    },
    
    "Add_Electrnic_noise": {
        "min_amount_salt": 0.005,
        "max_amount_salt": 0.05,
        "min_sigma_Gaussian_noise": 5,
        "max_sigma_Gaussian_noise": 15,
        "max_gaussian_Blur_sigma": 1,
        "Parasites_green_ch": 70,
        "Parasites_red_ch": 100,
        "Parasites_blue_ch": 40,
        "prob_glow": 0.5,
        "Prob_change_intensity": 0.5,
        
    },
    
    "Add_flurescent_noise": {
        "min_Flurescent_noise": 50,
        "max_Flurescent_noise": 100,
        "min_noisy_points": 100,
        "max_noisy_points": 1000,
    
    },

     "Path": {
        "csv_path": './fibers_coords',
        "glow_dir": './glow/',
        "output_dir_path": './output_images/',
        "Images_dir": './simulated_images/',
    
    },
   
}


Config_parameters = json.dumps(Noise_parameters)
print(Config_parameters)

# Using a JSON string
with open('Noise_parameters.json', 'w') as outfile:
    outfile.write(Config_parameters)