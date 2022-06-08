import json

Config_parameters = {
    "images_Number" : 50,
    
    "image_characteristics": {
        "image_width": 2048,
        "image_height": 2048,
        "dpi": 100,
    },
    
    "fiber_characteristics": {
        "min_Number_fibres": 5,
        "max_Number_fibres ": 50,
        "fiber_flectuation": 0.1,
        "fiber_min_lenght": 100,
        "dist_min": 30,
    },

     "Analog_characteristics": {
        "lmin_Analog": 40,
        "lmax_Analog": 150,
        "l_max_pair_analog": 300,
        "N_max_pair_analog": 3,
        "diff_l_analg": 50,
        "l_min_fibre_with_analog": 400,
    },
    

}


Config_parameters = json.dumps(Config_parameters)
print(Config_parameters)

# Using a JSON string
with open('Simulator_Config_parameters.json', 'w') as outfile:
    outfile.write(Config_parameters)
    

