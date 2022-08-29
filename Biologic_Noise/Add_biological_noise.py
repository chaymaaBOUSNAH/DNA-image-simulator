import numpy as np
import matplotlib.pyplot as plt
import math  
import random
import pandas as pd
import os
from pathlib import Path
import time
from PIL import Image
from .utils_bio import *
from .Draw_curves import *
from .Extract_curves_coords import extract_curves_coords


def Add_biological_noise(image, image_file, csv_path, total_noisy_fibers, min_Prob_perlage, max_Prob_perlage, min_N_pixels_perlage, max_lenght_perlage, pepper):
    start_time = time.time()
    assert np.amax(image)==255 and np.amin(image)==0,  'values are not in the range [0 255], normalise image values'
    raw, col, ch = image.shape

    csv_path = corresponding_csv_img(csv_path, image_file)
    file_fibers = os.path.basename(csv_path)

    
    assert len([file_fibers]) == 1, f'Either no file or multiple files found for the file {file_fibers}'

    # coord des fibres
    Fiber_data = pd.read_csv(csv_path)
    
    # extract curves coordonates:
    l1_u, l2_u, courbe_11, courbe_12, courbe_21,courbe_22 = extract_curves_coords(Fiber_data)
    
    image = image[:,:, 0:3]
    
    #Create figure 
    dpi = 100
    fig, ax = plt.subplots(1, 1, figsize=(raw, col), sharex=True, sharey=True, dpi = dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    fig.set_size_inches(raw/dpi, col/dpi)
    
    ax.imshow(image, cmap=plt.cm.gray, aspect='auto')
    
    
    
    for fiber in range(len(Fiber_data)):
        
        X1, X2 = Fiber_data['X1'][fiber], Fiber_data['X2'][fiber]
        Y1, Y2 = Fiber_data['Y1'][fiber], Fiber_data['Y2'][fiber]
        Width = Fiber_data['width'][fiber]
        Pente = Fiber_data['slop'][fiber]
        intercept = Fiber_data['b'][fiber]
        l_fibre = distance((X1, Y1), (X2, Y2))
        number_analogs = Fiber_data['N_analogs'][fiber]
        
        Prob_perlage = np.random.uniform(min_Prob_perlage, max_Prob_perlage)
        Perlage = random.choices(['true', 'false'], weights=[Prob_perlage, 1-Prob_perlage])

        
     
        # nombre de discontinuité dans une fibre -> proportionnel à la longeur de la fibre 
        # supposant pour 100 pixels --> 1 perlage
        num_perlage = int(l_fibre/min_N_pixels_perlage)
        
        for disc in range(num_perlage):
            
            # redessiner certaine morceaux de fibre bleues sur les fibres originales pour que certaine zone deviennent plus épaissent et pour faire du discontinuité seulement sur les analogues 
            
            if number_analogs !=0:
                l = np.random.uniform(5, 20)
                x__b = np.random.uniform(X1, X2-l)
                x__2b = x__b+l

                y__b = Pente*x__b + intercept
                y__2b  = Pente*x__b + intercept
                plt.plot((x__b, x__2b),(y__b, y__2b), color= 'b', linewidth=Width+0.5)
            
                
            if Perlage == ['true']:
                lenth = np.random.uniform(0.1, max_lenght_perlage)
                
                x__1 = np.random.uniform(X1, X2)
                #x__2 = np.random.uniform(x__1, x__1+lenth)
                x__2 = math.sqrt(lenth**2/(1+Pente**2))+x__1
       
                y__1 = Pente*x__1 + intercept
                y__2  = Pente*x__2 + intercept
       
                plt.plot((x__1, x__2),(y__1, y__2), color= 'black', linewidth=Width+0.5)
           
    
    Add_U_curve = ['true', 'false']
    Adding_curve = random.choices(Add_U_curve, weights=[0.6, 0.4])
    if np.size(l1_u) !=0 and Adding_curve==['true']:
        for k in range(len(l1_u)):
            
            p1, p2 = l1_u[k], l2_u[k]
      
            x_cercle, y_cercle = draw_cercle(p1, p2)
            plt.plot(y_cercle, x_cercle,  color = 'b', linewidth=Width)
    
    Add_Bezier_curve = ['true', 'false']
    Adding_Bezier_curve = random.choices(Add_Bezier_curve, weights=[0.7, 0.3])
    if np.size(courbe_11) !=0 and Adding_Bezier_curve==['true']:
        for k in range(len(courbe_11)):
            
            p_11, p_12 = courbe_11[k], courbe_12[k]
           
            curve1 = draw_bezier_curve(p_11, p_12, C = 'C1')

            plt.plot(
            	curve1[:, 0],   # x-coordinates.
            	curve1[:, 1],    # y-coordinates.
                color = 'b', linewidth=Width)
            
    if np.size(courbe_21) !=0 and Adding_Bezier_curve==['true']:
        for k in range(len(courbe_21)):
            
            p_21, p_22 = courbe_21[k], courbe_22[k]
          
            curve2 = draw_bezier_curve(p_21, p_22, C = 'C2')

            plt.plot(
            	curve2[:, 0],   # x-coordinates.
            	curve2[:, 1],    # y-coordinates.
                color = 'b', linewidth=Width)
            
            
    '''Add noisy fibers , fluerrescent noise, peper noise'''    
            
    for j in range(total_noisy_fibers):
        
        #bruit bilogique: fibres d'ADN et analogues
        p1 =  np.random.uniform(1, 20)       
        # coordonnées x des fibres d'ADN
        x1 = np.random.uniform(0, raw) 
        #lmin pour ne pas avoir des fibre trop petites(qui ressemblent au bruit) 
        x2 = x1+p1
        # coordonnées y des fibres d'ADN
        y1 = np.random.uniform(0, col) 
        #déterminer l'intercept de la droite
        # la pente est la meme que les fibres (à partor des fivhier csv)
        pente = Fiber_data['slop'][0]
        b = y1 - pente*x1
        # calculer y2 de la meme fibre
        y2 = pente*x2 + b
        
        # morceaux des fibres des analogues comme bruit
        noise_colors = ['b', 'aqua', 'magenta']
        noise_color = np.random.choice(noise_colors, 1, p = [0.7, 0.15, 0.15])
        linewidth = np.random.randint(2, 8)
        plt.plot((x1, x2),(y1, y2), color= noise_color[0], linewidth=linewidth)
        

        
    #colors = ['#32CD32', '#FF34B3']
    #color = np.random.choice(colors, 1, p = [0.6, 0.4])
    for j in range(pepper):
        x = np.random.uniform(0, raw)
        y = np.random.uniform(0, col)
        size = np.random.uniform(1, 7)
        #colors = ['#00FF00', 'b']
        #color = np.random.choice(colors, 1, p = [0.6, 0.4])
        markers = ['s', 'o']
        markr = np.random.choice(markers, 1, p=[0.6, 0.4])
        plt.scatter(x, y, s=size, c='#00FF00', marker=markr[0])
        plt.scatter(x, y, s=size, c='#0000FF', marker=markr[0])
    
 
    """    
    for noisy_dust in range(pepper):   
        # autre bruit : poussière
        x = np.random.uniform(0, 2048)
        y = np.random.uniform(0, 2048)
        colors = ['#32CD32', '#FF34B3']
        color = random.choices(colors, weights=[0.8, 0.2])
        alpha_value = 0.3
        
        n_point =  np.random.randint(10, 30)
        s = np.random.uniform(0.01, 7)
         
        for n in range(1, n_point+1):
            
            if n<5:
                plt.scatter(x,y, marker='o', c = 'w', s = s*n, alpha = alpha_value/n)
            else:
                plt.scatter(x,y, marker='o',color = color, s = s*n , alpha = alpha_value/(n*1.5))  
      """          
    
    # plot randomly a line
    proba_noisy_line = random.choices(['true', 'false'], weights=(0.01, 0.9))
    if proba_noisy_line == ['true']:
        l_x = np.random.randint(10, raw)
        x1_line = np.random.uniform(0, raw-l_x)
        x2_line = x1_line + l_x
        
        l_y = random.randint(1000, col)
        y1_line = np.random.uniform(0, col-l_y)
        y2_line  = y1_line+ l_y
        
        colors = ['#800080', '#FF34B3', '#D02090', '#C71585']
        color = random.choices(colors, weights=[0.3, 0.3, 0.2, 0.2])
        
        W = random.randint(20, 50)
        plt.plot((x1_line, x2_line), (y1_line, y2_line), color= color[0], linewidth=W, alpha = 0.9)
    
    
    ax.set_xlim((0, image.shape[1]))
    ax.set_ylim((image.shape[0], 0))
    
    
    ax.set_xlim((0, image.shape[1]))
    ax.set_ylim((image.shape[0], 0))
    
    print('Biological noise', "--- %s seconds ---" % (time.time() - start_time))
    
    
    return canvas2rgb_array(fig.canvas) 



