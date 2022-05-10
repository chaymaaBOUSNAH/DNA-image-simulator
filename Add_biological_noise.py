import numpy as np
import matplotlib.pyplot as plt
import math  
import tifffile
import random
import pandas as pd
import numpy as np
import os
#import cv2
import matplotlib as mpl

from os import walk
from os.path import join
from matplotlib.patches import Arc

from matplotlib import image
from matplotlib import cm
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.filters import threshold_mean, threshold_triangle

from Bezier_curve import Bezier

mpl.rc('figure', max_open_warning = 0)


import re 

def sorted_file( l ): 
    
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


# fonction ppou lire tous les fichier csv d'un dossier et les enregister dans une liste
def All_csv(files_path):    
    list_csv = []
    for (dirpath, dirnames, filenames) in walk(files_path):
        for csv_name in sorted_file(filenames):
            csv_path = join(files_path, csv_name)
            
            csv = pd.read_csv(csv_path)
            #print('csv', csv_name, csv)
            list_csv.append(csv)
    return list_csv
    

#lire les fichier csv
files_path1 = './create_images/L_U1/'
all_lines_U1 = All_csv(files_path1)

files_path2 = './create_images/L_U2/'
all_lines_U2 = All_csv(files_path2)
        

files_path3 = './create_images/L_C11/'    
all_lines_C11 = All_csv(files_path3)

files_path4 = './create_images/L_C12/' 
all_lines_C12  = All_csv(files_path4)

files_path5 = './create_images/L_C21/' 
all_lines_C21  = All_csv(files_path5)

files_path6 = './create_images/L_C22/' 
all_lines_C22  = All_csv(files_path6)


def hanging_line(point1, point2):
    import numpy as np
    import sympy as sy

    a = (point2[0]-point1[0])/(sy.cosh(point2[1])-sy.cosh(point1[1]))
    b = point1[0]-a*sy.cosh(point1[1])
    x = np.linspace(point1[1], point2[1], 20000)
    #y = a*sy.cosh(x)+b
    
    
    y = []
    for i in x:
        y.append(a*sy.cosh(i)+b)


    return (x,y)

# dessiner du bruit en forme de u


def draw_cercle(P1, P2):
    x1, y1 = P1[0], P1[1]
    x2, y2 = P2[0], P2[1]
    diameter = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
    r = diameter / 2
    xC, yC = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    
    x = np.linspace(x1, x2, 2000)
    
    # changer de signe pour faire le cercle dans l'autre sens
    y = [yC-(r**2 - (value-xC)**2)**0.5 for value in x] 
    
    return x, y 

def draw_bezier_curve(point1, point2, C):
    
    x_1, y_1 = point1[0], point1[1]
    x_2, y_2 = point2[0], point2[1]
    
    l = abs(y_2-y_1)
    L = abs(x_2-x_1)
    
    #Définir deux points pour la courbe de bézier 
    # sens 1
    if C == 'C1':
        i_1_x, i_1_y = x_1-L/3, y_1
        i_2_x, i_2_y = x_2+L/3, y_2
    # sens 2
    if C == 'C2':
        i_1_x, i_1_y = x_2+L/3, y_1
        i_2_x, i_2_y = x_1-L/3, y_2
    
    points = np.array([[x_1, y_1], [i_1_x , i_1_y], [i_2_x, i_2_y], [x_2, y_2]])
    t_points = np.arange(0, 1, 0.01)

    curve = Bezier.Curve(t_points, points)
    
    return curve


N_Biologic_noise = np.random.randint(5, 40) # le nombre possible de fibres(bruit) dans une image
noisy_fibers = np.random.randint(20, 50)
noisy_dust = np.random.randint(50, 200)
perlage = np.random.randint(6000, 12000)
m = np.random.uniform(0, 0.01)




images_path = './create_images/noisy_fibers/'
for (dirpath, dirnames, filenames) in walk(images_path):
    
    file_index = 0
    for image_file in sorted_file(filenames):
        
        image_path = join(images_path, image_file)
        
        imag = plt.imread(image_path)
        """
        csv1 = all_lines_U1[file_index]
        csv2 = all_lines_U2[file_index]
        
        
        csv3 = all_lines_C11[file_index]         
        csv4 = all_lines_C12[file_index]
        csv5 = all_lines_C21[file_index]         
        csv6 = all_lines_C22[file_index]
        """
        image = imag[:,:, 0:3]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        fig.set_size_inches(2048/100, 2048/100)
        
        ax.imshow(image, cmap=plt.cm.gray, aspect='auto')
        """
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
                
        for j in range(noisy_fibers):
            
            #bruit bilogique: fibres d'ADN et analogues
            p1 =  np.random.uniform(1, 20)       
            # coordonnées x des fibres d'ADN
            x1 = np.random.uniform(0, 2048) 
            #lmin pour ne pas avoir des fibre trop petites(qui ressemblent au bruit) 
            x2 = x1+p1
            # coordonnées y des fibres d'ADN
            y1 = np.random.uniform(0, 2048) 
            #déterminer l'intercept de la droite
            b = y1 - m*x1
            # calculer y2 de la meme fibre
            y2 = m*x2 + b
            
            # morceaux des fibres des analogues comme bruit
            noise_colors = ['b', 'aqua', 'magenta']
            noise_color = np.random.choice(noise_colors, 1, p = [0.8, 0.1, 0.1])
            linewidth = np.random.randint(2, 8)
            plt.plot((x1, x2),(y1, y2), color= noise_color[0], linewidth=linewidth)
            
        for noise in range(N_Biologic_noise):     
            a = np.random.uniform(0, 2048)
            b = np.random.uniform(0, 2048)
            s = np.random.randint(1, 20)
            plt.scatter(a, b, marker='o', c = 'b', s = s)
        """    
        for noisy_dust in range(noisy_dust):   
            # autre bruit : poussière
            x = np.random.uniform(0, 2048)
            y = np.random.uniform(0, 2048)
            colors = ['#32CD32', '#FF00FF']
            color = random.choices(colors, weights=[0.6, 0.4])
            alpha_value = 0.3
            
            n_point =  np.random.randint(10, 50)
            s = np.random.uniform(0.1, 30)
                
            for n in range(1, n_point+1):
                if n<5:
                    plt.scatter(x,y, marker='o', c = 'w', s = s*n, alpha = alpha_value/n)
                else:
                    plt.scatter(x,y, marker='o',color = color, s = s*n , alpha = alpha_value/(n*1.5))
                 
        colors = ['#32CD32', '#FF34B3']
        color = np.random.choice(colors, 1, p = [0.8, 0.2])
        for j in range(perlage):
            x = np.random.uniform(0, 2048)
            y = np.random.uniform(0, 2048)
            size = np.random.uniform(1, 10)
            #colors = ['#32CD32', '#FF34B3']
            #color = np.random.choice(colors, 1, p = [0.7, 0.3])
            markers = ['s', 'o']
            markr = np.random.choice(markers, 1, p=[0.6, 0.4])
            # BLUE #0000FF
            # magenta #FF00FF: (255,0,255) # Aqua 00FFFF:(0,255,255)
            # GREENYELLOW #ADFF2F (173,255,47)
            # olivedrab1 #C0FF3E (192,255,62)
            # maroon1 #FF34B3 (255,52,179)
            # limegreen #32CD32 RGB(50,205,50)
            
            plt.scatter(x, y, s=size, c=color[0], marker=markr[0])    
            
            '''
            x_b = y_b = np.random.uniform(0, 2048)
            y_b = np.random.uniform(0, 2048)
            siz = np.random.uniform(2, 10)
            plt.scatter(x_b, y_b, s = siz, c='black', marker=markr[0])             
            '''
         
        ax.set_xlim((0, image.shape[1]))
        ax.set_ylim((image.shape[0], 0))
        #ax[1].set_title('draw lines')
         
               
        plt.savefig('./create_images/noisy/image_'+str(file_index), bbox_inches='tight', pad_inches=0, dpi = 100)

        
        file_index +=1

