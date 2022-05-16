import numpy as np
import matplotlib.pyplot as plt
import math  
import random
import pandas as pd
#import cv2
import matplotlib as mpl
import json
from utils import distance
mpl.rc('figure', max_open_warning = 0)


"load parameters from json file "

with open('Config_parameters.json') as json_file:
    data = json.load(json_file)
    print('generator_parameters:', data)
    
"""paramètres fichier de configuration:"""
    
# Nombre d'image à générer
N_images = data['images_Number']

# image_characteristics
image_characteristics = data['image_characteristics']
image_width = image_characteristics['image_width']
image_height = image_characteristics['image_height']
#Dots per inches (dpi) determines how many pixels the figure comprises
dpi = image_characteristics['dpi']

# fiber_characteristics
fiber_characteristics = data['fiber_characteristics']
#Nombre minimal et maximal de fibres à générer/image
min_N_fibres = fiber_characteristics['min_Number_fibres']
max_N_fibres = fiber_characteristics['max_Number_fibres']
#fluctuation de la pente des fibres ( entre 0 et 0,2)
delta = fiber_characteristics['fiber_flectuation'] 
# longueur min d'un fibre en pixel
l_min = fiber_characteristics['fiber_min_lenght']
# distance minimale entre 2 fibres linéaires selon l'axe x
dist_min = fiber_characteristics['dist_min'] 

# Analog_characteristics
Analog_characteristics = data['Analog_characteristics']
lmin_Analog = Analog_characteristics['lmin_Analog']
lmax_Analog = Analog_characteristics['lmax_Analog']
lmax = Analog_characteristics['l_max_pair_analog'] 
#nombre max de paire d'analogue sur une fibre
Max_analog = Analog_characteristics['N_max_pair_analog'] 
# les analogues doivent etre plus petites que les fibres 
# la différence max de longeur entre une paire d'analogues dif_analg
dif_analg = Analog_characteristics['diff_l_analg'] 
# la longueur min d'une fibre qui peut avoir des analogues min_fibre
min_fibre = Analog_characteristics['l_min_fibre_with_analog'] 

# curves_characteristics
curves_characteristics = data['curves_characteristics']
# les distan,ces en x et y entre chaque deux fibres qui forment des courbes de bezier
distU_X= curves_characteristics['distU_X']
distU_Y= curves_characteristics['distU_Y']

distx_min  = curves_characteristics['distx_min']
disty_min = curves_characteristics['disty_min']
distx_max = curves_characteristics['distx_max']
disty_max = curves_characteristics['disty_max']

# la pente des fibres sur toute les image

for i in range(N_images):
    fiber_width = np.random.randint(4, 6)
    
    m = np.random.uniform(0, delta)
    pente = ['normal', 'decale']
    choix_pente = random.choices(pente, weights=[0.7, 0.3])
    
    # P1 et P2 sont des listes contenants les coordonnées x et y des 2 points de deux fibre
    X1 = [] 
    X2 = []
    Y1 = [] 
    Y2 = []
    # l1_u coord (X2,Y2) des fibres qui peuvent etres en forme de U après
    # l2_u = [] coord (X2,Y2) des fibres dont la distance des x avec le fibre précédent est inf à 10
    l1_u = []
    l2_u = []
    
    Pente = []
    Intercept = []
    width = []
    
    last_fibre_x2 = []
    last_fibre_y2 = []
    
    last_fibre_x1 = []
    last_fibre_y1 = []
    
    courbe_11 = []
    courbe_12 = []
    
    courbe_21 = []
    courbe_22 = []
    
    # générer un nombre aléatoire de fibres/image
    N = np.random.randint(min_N_fibres, max_N_fibres) 
    
    # générer des cordonnées aléatoire selon l'axe y (nombre de coordonnées générées = nombres de fibres choisis)
    data = np.random.randint(0, image_height, N)
    data = np.sort(data)
          
    #créer une image vide (matrice de 0)
    Image = np.zeros((image_height, image_width))
    #paramètres de l'image
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    # dpi=100 will contain image_heigh/dpi
    fig.set_size_inches(image_width/dpi, image_width/dpi)
    
    ax.imshow(Image, cmap=plt.cm.gray, aspect='auto')
    
    for j in range(N):
        diff_width = np.random.uniform(0, 0.1)
        fiber_width  = fiber_width + diff_width
        
        # changer la pente sur l'image
        if choix_pente==['decale']:
            m = m+0.001
         
        # coordonnées x des fibres d'ADN
        x1 = np.random.uniform(0, image_width) 
        #lmin pour ne pas avoir des fibre trop petites(qui ressemblent au bruit) 
        x2 = np.random.uniform(x1+l_min, image_width) 
        # coordonnées y des fibres d'ADN
        y1 = data[j]
        #déterminer l'intercept de la droite
        b = y1 - m*x1
        # calculer y2 de la meme fibre
        y2 = m*x2 + b
        
        # P1 et P2 de la fibre:
        p1 = (x1, y1)
        p2 = (x2, y2)
        
        # Enregistrer les coordonnérs des fibres 
        X1.append(x1)
        X2.append(x2)
        Y1.append(y1)
        Y2.append(y2)
        
        width.append(fiber_width)
        Pente.append(m)
        Intercept.append(b)
 
        # calculer la longueur des fibres 
        l_fibre = distance(p1, p2)
        # dessiner les fibres
        plt.plot((x1, x2),(y1, y2), 'b', linewidth=fiber_width)
        
        # enregister les coordonnées avecx lequelles on va faire la forme U
        last_fibre_x2.append(x2)
        last_fibre_y2.append(y2)
        
        # enregister les coordonnées avecx lequelles on va faire une courbe entre deux fibres
        last_fibre_x1.append(x1)
        last_fibre_y1.append(y1)
        # distance entre une fibre et la fibre précédente
        dist_U_fibre = distance((last_fibre_x2[j], last_fibre_y2[j]), (last_fibre_x2[j-1], last_fibre_y2[j-1]))

        distx_u = abs(last_fibre_x1[j-1]-last_fibre_x1[j])
        disty_u = abs(last_fibre_y1[j-1]-last_fibre_y1[j])
        
        if distx_u<distU_X and disty_u<distU_Y:    
            l1_u.append([last_fibre_x1[j-1], last_fibre_y1[j-1]])
            l2_u.append([last_fibre_x1[j], last_fibre_y1[j]])
      
        
        # enregister les lignes sur les quel on va ajouter un autre type de bruit
        distx_C1 = abs(last_fibre_x1[j]-last_fibre_x1[j-1])
        disty_C1 = abs(last_fibre_y1[j]-last_fibre_y1[j-1])
        if last_fibre_x2[j]<last_fibre_x1[j-1] and distx_C1 <distx_max and distx_C1>distx_min and disty_C1<disty_max:           
            courbe_11.append([last_fibre_x1[j-1], last_fibre_y1[j-1]])
            courbe_12.append([last_fibre_x1[j], last_fibre_y1[j]])
        
        distx_C2 = abs(last_fibre_x2[j-1]-last_fibre_x1[j])
        disty_C2 = abs(last_fibre_y2[j-1]-last_fibre_y1[j])
        if last_fibre_x2[j-1]<last_fibre_x1[j] and distx_C2 <distx_max and distx_C2>distx_min and disty_C2<disty_max:
            courbe_21.append([last_fibre_x2[j-1], last_fibre_y2[j-1]])
            courbe_22.append([last_fibre_x1[j], last_fibre_y1[j]])
    
        # Dessiner les analoguers de nucléotide sur les fibres
        
        # nombre aléatoire de paire d'analogue sur une fibre
        n_pair_analog = np.random.randint(0, Max_analog)
        
        for k in range(n_pair_analog):
            # coordonnées du premier analogue
            Analog1_x1 =  np.random.uniform(x1, x2-lmax) 
            Analog1_x2 = np.random.uniform(Analog1_x1+lmin_Analog, Analog1_x1+lmax_Analog)
            
            # Calculer à partir de l'équation de droite des premiers fibres
            Analog1_y1 =  m*Analog1_x1+b
            Analog1_y2 = m*Analog1_x2+b
            
            P1_Analog1 = (Analog1_x1, Analog1_y1)
            P2_Analog1 = (Analog1_x2, Analog1_y2)
            
            l_Analog1 = distance((Analog1_x1, Analog1_y1), (Analog1_x2, Analog1_y2))
            
            # Analog2
            Analog2_x1 = Analog1_x2
            Analog2_x2 = np.random.uniform(Analog2_x1+lmin_Analog, x2)
            
            Analog2_y1 = Analog1_y2
            Analog2_y2 = m*Analog2_x2+b
            
            P1_Analog2 = (Analog2_x1, Analog2_y1)
            P2_Analog2 = (Analog2_x2, Analog2_y2)
            
            l_Analog2 = distance((Analog2_x1, Analog2_y1), (Analog2_x2, Analog2_y2))
            l_analog = l_Analog1+l_Analog2
            
            
            if l_fibre>min_fibre and abs(l_Analog2-l_Analog1)<dif_analg and l_fibre>l_analog:  
                #magenta FF00FF: (255,0,255) # Aqua 00FFFF:(0,255,255)
                colors = np.array(['#FF00FF', '#00FFFF'])
                color1 = random.choice(colors)
                index =  np.where(colors != color1)
                color2 = colors[index[0][0]]
    
                
                # dessiner clu sur l'image 
                plt.plot((Analog1_x1, Analog1_x2),(Analog1_y1,Analog1_y2), color = color1, linewidth=fiber_width)
            
                # dessiner cldu sur l'image
                plt.plot((Analog2_x1, Analog2_x2),(Analog2_y1, Analog2_y2), color = color2, linewidth=fiber_width)
                
        

    ax.set_xlim((0, Image.shape[1]))
    ax.set_ylim((Image.shape[0], 0))
    
    plt.savefig('./images/image_'+str(i)+'_mask', bbox_inches='tight', pad_inches=0, dpi=dpi)
    
    # enregister les coordonées des fibres de chaque image dans un fichier csv
    
    #fibre_data = np.hstack((X1, Y1, X2, Y2, Pente, width))
    fibre = pd.DataFrame([X1, Y1, X2, Y2, Pente, Intercept, width],  dtype='f')
    fibre = fibre.transpose() 
    fibre.to_csv('./fibres_data/image_'+str(i)+'_mask', header=['X1', 'Y1', 'X2', 'Y2', 'slop', 'b', 'width'], index=False)
    

    line1_xy = pd.DataFrame(l1_u, columns=['X', 'Y'],  dtype='f')
    line1_xy.to_csv('./Curves_data/L_U1/image_'+str(i)+'_mask', index=False)
    
    line2_xy = pd.DataFrame(l2_u, columns=['X', 'Y'],dtype='f')
    line2_xy.to_csv('./Curves_data/L_U2/image_'+str(i)+'_mask', index=False)
    
    courbe_11 = pd.DataFrame(courbe_11, columns=['X', 'Y'],  dtype='f')
    courbe_11.to_csv('./Curves_data/L_C11/image_'+str(i)+'_mask', index=False)
    
    courbe_12 = pd.DataFrame(courbe_12, columns=['X', 'Y'],  dtype='f')
    courbe_12.to_csv('./Curves_data/L_C12/image_'+str(i)+'_mask', index=False)
    
    courbe_21 = pd.DataFrame(courbe_21, columns=['X', 'Y'],  dtype='f')
    courbe_21.to_csv('./Curves_data/L_C21/image_'+str(i)+'_mask', index=False)
    
    courbe_22 = pd.DataFrame(courbe_22, columns=['X', 'Y'],  dtype='f')
    courbe_22.to_csv('./Curves_data/L_C22/image_'+str(i)+'_mask', index=False)
    
    
