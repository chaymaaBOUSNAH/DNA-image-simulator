import numpy as np
import matplotlib.pyplot as plt
import math  
import random
import pandas as pd
import os
from PIL import Image
from utils import canvas2rgb_array, distance
from utils import sorted_file
from Draw_curves import extract_curves_coords, draw_cercle, draw_bezier_curve
from pathlib import Path
import time

def Add_biological_noise(image, image_file, csv_path, total_noisy_fibers, min_Prob_perlage, max_Prob_perlage, min_N_pixels_perlage, max_lenght_perlage, pepper):
    start_time = time.time()
    assert np.amax(image)==255 and np.amin(image)==0,  'values are not in the range [0 255], normalise image values'
    
    image_name = image_file.split('.')
    name = image_name[0]
    
    
    # File loading
    
    # coord des fibres
    File_Fibers = list(Path(csv_path).glob(name))[0]
    Fiber_data = pd.read_csv(File_Fibers)
    
    #assert len(File_Fibers) == 1 , f'Either no file or multiple files found for the one of the files fibers coordonates{File_Fibers}'
    
    # extract curves coordonates:
    l1_u, l2_u, courbe_11, courbe_12, courbe_21,courbe_22 = extract_curves_coords(Fiber_data)
    
    image = image[:,:, 0:3]
    
    #Create figure 
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True, dpi = 100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    fig.set_size_inches(2048/100, 2048/100)
    
    ax.imshow(image, cmap=plt.cm.gray, aspect='auto')
    
    Add_U_curve = ['true', 'false']
    Adding_curve = random.choices(Add_U_curve, weights=[0.7, 0.3])
    if np.size(l1_u) !=0 and Adding_curve==['true']:
        for k in range(len(l1_u)):
            
            p1, p2 = l1_u[k], l2_u[k]
      
            x_cercle, y_cercle = draw_cercle(p1, p2)
            plt.plot(y_cercle, x_cercle,  color = 'b', linewidth=5)
    
    Add_Bezier_curve = ['true', 'false']
    Adding_Bezier_curve = random.choices(Add_Bezier_curve, weights=[0.7, 0.3])
    if np.size(courbe_11) !=0 and Adding_Bezier_curve==['true']:
        for k in range(len(courbe_11)):
            
            p_11, p_12 = courbe_11[k], courbe_12[k]
           
            curve1 = draw_bezier_curve(p_11, p_12, C = 'C1')

            plt.plot(
            	curve1[:, 0],   # x-coordinates.
            	curve1[:, 1],    # y-coordinates.
                color = 'b', linewidth=5)
            
    if np.size(courbe_21) !=0 and Adding_Bezier_curve==['true']:
        for k in range(len(courbe_21)):
            
            p_21, p_22 = courbe_21[k], courbe_22[k]
          
            curve2 = draw_bezier_curve(p_21, p_22, C = 'C2')

            plt.plot(
            	curve2[:, 0],   # x-coordinates.
            	curve2[:, 1],    # y-coordinates.
                color = 'b', linewidth=5)
            
            
    '''Add noisy fibers , fluerrescent noise, peper noise'''    
            
    for j in range(total_noisy_fibers):
        
        #bruit bilogique: fibres d'ADN et analogues
        p1 =  np.random.uniform(1, 20)       
        # coordonnées x des fibres d'ADN
        x1 = np.random.uniform(0, 2048) 
        #lmin pour ne pas avoir des fibre trop petites(qui ressemblent au bruit) 
        x2 = x1+p1
        # coordonnées y des fibres d'ADN
        y1 = np.random.uniform(0, 2048) 
        #déterminer l'intercept de la droite
        # la pente est la meme que les fibres (à partor des fivhier csv)
        pente = Fiber_data['slop'][0]
        b = y1 - pente*x1
        # calculer y2 de la meme fibre
        y2 = pente*x2 + b
        
        # morceaux des fibres des analogues comme bruit
        noise_colors = ['b', 'aqua', 'magenta']
        noise_color = np.random.choice(noise_colors, 1, p = [0.8, 0.1, 0.1])
        linewidth = np.random.randint(2, 8)
        plt.plot((x1, x2),(y1, y2), color= noise_color[0], linewidth=linewidth)
        
    for fiber in range(len(Fiber_data)):
        
        X1, X2 = Fiber_data['X1'][fiber], Fiber_data['X2'][fiber]
        Y1, Y2 = Fiber_data['Y1'][fiber], Fiber_data['Y2'][fiber]
        Width = Fiber_data['width'][fiber]
        Pente = Fiber_data['slop'][fiber]
        intercept = Fiber_data['b'][fiber]
        l_fibre = distance((X1, Y1), (X2, Y2))
        
        Prob_perlage = np.random.uniform(min_Prob_perlage, max_Prob_perlage)
        Perlage = random.choices(['true', 'false'], weights=[Prob_perlage, 1-Prob_perlage])
        
        if Perlage == ['true']:
           
            # nombre de discontinuité dans une fibre -> proportionnel à la longeur de la fibre 
            # supposant pour 100 pixels --> 1 perlage
            num_perlage = int(l_fibre/min_N_pixels_perlage)
            
            for disc in range(num_perlage):
                lenth = np.random.randint(1, max_lenght_perlage)
                
                x__1 = np.random.uniform(X1, X2)
                #x__2 = np.random.uniform(x__1, x__1+lenth)
                x__2 = math.sqrt(lenth**2/(1+Pente**2))+x__1
       
                y__1 = Pente*x__1 + intercept
                y__2  = Pente*x__2 + intercept
            
                plt.plot((x__1, x__2),(y__1, y__2), color= 'black', linewidth=Width+0.5)            
        
    #colors = ['#32CD32', '#FF34B3']
    #color = np.random.choice(colors, 1, p = [0.6, 0.4])
    for j in range(pepper):
        x = np.random.uniform(0, 2048)
        y = np.random.uniform(0, 2048)
        size = np.random.uniform(1, 7)
        #colors = ['#00FF00', 'b']
        #color = np.random.choice(colors, 1, p = [0.6, 0.4])
        markers = ['s', 'o']
        markr = np.random.choice(markers, 1, p=[0.6, 0.4])
        plt.scatter(x, y, s=size, c='#00FF00', marker=markr[0])    
    for j in range(pepper):
        x = np.random.uniform(0, 2048)
        y = np.random.uniform(0, 2048)
        size = np.random.uniform(6, 20)
        markers = ['s', 'o']
        markr = np.random.choice(markers, 1, p=[0.6, 0.4])
        
        plt.scatter(x, y, s=size, c='#0000FF', marker=markr[0])   
        
        
    ax.set_xlim((0, image.shape[1]))
    ax.set_ylim((image.shape[0], 0))
    
    
    ax.set_xlim((0, image.shape[1]))
    ax.set_ylim((image.shape[0], 0))
    print('Biological noise', "--- %s seconds ---" % (time.time() - start_time))  
    return canvas2rgb_array(fig.canvas) 
# le nombre de discontinuité à effecter sur chaque fibre  
#Add_biological_noise

'''
# tester la fonction

total_noisy_fibers = np.random.randint(20, 50)
noisy_dust = np.random.randint(50, 200)
perlage = np.random.randint(500, 2000)
prob = 1

# la longueur min de pixel pour avoir un perlage 
min_N_pixels_perlage = 100
max_lenght_perlage = 80

#lire les fichier csv des coordonnées des courbes

csv_path = './fibers_coords' 

pepper = 1000   

image_path = './simulated_images/image_0_mask.png'

fig = Add_biological_noise(image_path, csv_path, total_noisy_fibers, prob, min_N_pixels_perlage, max_lenght_perlage, pepper)
img = canvas2rgb_array(fig.canvas)
pil_image=Image.fromarray(img)
pil_image.show()

'''
