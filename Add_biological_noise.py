import numpy as np
import matplotlib.pyplot as plt
import math  
import random
import pandas as pd
import matplotlib as mpl
from os import walk
from os.path import join
mpl.rc('figure', max_open_warning = 0)
import os
from PIL import Image
from utils import canvas2rgb_array, distance
from utils import sorted_file
from Draw_curves import draw_cercle, draw_bezier_curve




def Add_biological_noise(image_path, All_csv_path, total_noisy_fibers, Prob_perlage, min_N_pixels_perlage, max_lenght_perlage, pepper):
    
    image_head, image_file = os.path.split(image_path)
    image_name = image_file.split('.')
    name = image_name[0]
    
    imag = plt.imread(image_path)
    
    # curve U
    csv1 = pd.read_csv(All_csv_path[0]+name)
    csv2 = pd.read_csv(All_csv_path[1]+name)
    
    # Bezier curves
    csv3 = pd.read_csv(All_csv_path[2]+name)
    csv4 = pd.read_csv(All_csv_path[3]+name)
    csv5 = pd.read_csv(All_csv_path[4]+name)
    csv6 = pd.read_csv(All_csv_path[5]+name)
    
    # coord des fibres
    Fiber_data = pd.read_csv(All_csv_path[6]+name)
    

    image = imag[:,:, 0:3]
    
    #Create figure 
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True, dpi = 100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    fig.set_size_inches(2048/100, 2048/100)
    
    ax.imshow(image, cmap=plt.cm.gray, aspect='auto')
    
    Add_U_curve = ['true', 'false']
    Adding_curve = random.choices(Add_U_curve, weights=[0.7, 0.3])
    if np.size(csv1) !=0 and Adding_curve==['true']:
        for k in range(len(csv1)):
            
            x1, x2 = csv1['X'][k], csv2['X'][k]
            y1, y2 = csv1['Y'][k], csv2['Y'][k]
      
            x_cercle, y_cercle = draw_cercle((y2, x2), (y1,x1))
            plt.plot(y_cercle, x_cercle,  color = 'b', linewidth=5)
    
    Add_Bezier_curve = ['true', 'false']
    Adding_Bezier_curve = random.choices(Add_Bezier_curve, weights=[0.7, 0.3])
    if np.size(csv3) !=0 and Adding_Bezier_curve==['true']:
        for k in range(len(csv3)):
            
            x_11, x_12 = csv3['X'][k], csv4['X'][k]
            y_11, y_12 = csv3['Y'][k], csv4['Y'][k]
          
            curve1 = draw_bezier_curve((x_11, y_11), (x_12, y_12), C = 'C1')

            plt.plot(
            	curve1[:, 0],   # x-coordinates.
            	curve1[:, 1],    # y-coordinates.
                color = 'b', linewidth=5)
            
    if np.size(csv5) !=0 and Adding_Bezier_curve==['true']:
        for k in range(len(csv5)):
            
            x_21, x_22 = csv5['X'][k], csv6['X'][k]
            y_21, y_22 = csv5['Y'][k], csv6['Y'][k]
          
            curve2 = draw_bezier_curve((x_21, y_21), (x_22, y_22), C = 'C2')

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
        
        Avec_perlage = ['true', 'false']
        Perlage = random.choices(Avec_perlage, weights=[Prob_perlage, 1-Prob_perlage])
        
        if Perlage == ['true']:
            # num perlage proportionnel à la longeur de la fibre 
            # supposant pour 100 pixels --> 1 perlage
  
            num_perlage = int(l_fibre/min_N_pixels_perlage)
            
            for disc in range(num_perlage):
                lenth = np.random.uniform(1, max_lenght_perlage)
                
                x__1 = np.random.uniform(X1, X2)
                #x__2 = np.random.uniform(x__1, x__1+lenth)
                x__2 = math.sqrt(lenth/(1+Pente**2))+x__1
       
                y__1 = Pente*x__1 + intercept
                y__2  = Pente*x__2 + intercept
            
                plt.plot((x__1, x__2),(y__1, y__2), color= 'black', linewidth=Width+0.5)            
        
    #colors = ['#32CD32', '#FF34B3']
    #color = np.random.choice(colors, 1, p = [0.6, 0.4])
    for j in range(pepper):
        x = np.random.uniform(0, 2048)
        y = np.random.uniform(0, 2048)
        size = np.random.uniform(0.1, 10)
        colors = ['#00FF00', 'r', 'b']
        color = np.random.choice(colors, 1, p = [0.5, 0.2, 0.3])
        markers = ['s', 'o']
        markr = np.random.choice(markers, 1, p=[0.6, 0.4])
        # BLUE #0000FF
        # magenta #FF00FF: (255,0,255) # Aqua 00FFFF:(0,255,255)
        # GREENYELLOW #ADFF2F (173,255,47)
        # olivedrab1 #C0FF3E (192,255,62)
        # maroon1 #FF34B3 (255,52,179)
        # limegreen #32CD32 RGB(50,205,50)
        
        plt.scatter(x, y, s=size, c=color[0], marker=markr[0])    
    ax.set_xlim((0, image.shape[1]))
    ax.set_ylim((image.shape[0], 0))
    
    
    ax.set_xlim((0, image.shape[1]))
    ax.set_ylim((image.shape[0], 0))
    
    return fig
# le nombre de discontinuité à effecter sur chaque fibre  
#Add_biological_noise


'''
total_noisy_fibers = np.random.randint(20, 50)
noisy_dust = np.random.randint(50, 200)
noisy_points = np.random.randint(20000, 40000)
perlage = np.random.randint(500, 2000)
prob = 1

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

image_path = './images/image_6_mask.png'
All_csv_path = [files_path1, files_path2, files_path3, files_path4, files_path5, files_path6, files_path7]

fig = Add_biological_noise(image_path, All_csv_path, total_noisy_fibers, prob, min_N_pixels_perlage, max_lenght_perlage)
img = canvas2rgb_array(fig.canvas)
pil_image=Image.fromarray(img)
pil_image.show()
pil_image.save('./Essai/image_6_.png')

'''