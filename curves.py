import numpy as np
import pandas as pd
from utils import distance

def extract_curves_coords(Fibers_coords):
    
    # curves_characteristics
    # les distan,ces en x et y entre chaque deux fibres qui forment des courbes de bezier
    distU_X= 10
    distU_Y= 300
    
    distx_min  = 30
    disty_min = 10
    distx_max = 100
    disty_max = 100
    
    
    
    # l1_u coord (X2,Y2) des fibres qui peuvent etres en forme de U après
    # l2_u = [] coord (X2,Y2) des fibres dont la distance des x avec le fibre précédent est inf à 10
    l1_u = []
    l2_u = []
    
    courbe_11 = []
    courbe_12 = []
    
    courbe_21 = []
    courbe_22 = []
    
    
    
    
    for j in range(len(Fibers_coords)):
        
        X1, X2 = Fibers_coords['X1'], Fibers_coords['X2']
        Y1, Y2 = Fibers_coords['Y1'], Fibers_coords['Y2']
        
        if j>0:
     
            
            distx_u = abs(X1[j-1]-X1[j])
            disty_u = abs(Y1[j-1]-Y1[j])
            
            if distx_u<distU_X and disty_u<distU_Y:    
                l1_u.append([X1[j-1], Y1[j-1]])
                l2_u.append([X1[j], Y1[j]])
            
            
            # enregister les lignes sur les quel on va ajouter un autre type de bruit
            distx_C1 = abs(X1[j]-X1[j-1])
            disty_C1 = abs(Y1[j]-Y1[j-1])
            if X2[j]<X1[j-1] and distx_C1 <distx_max and distx_C1>distx_min and disty_C1<disty_max:           
                courbe_11.append([X1[j-1], Y1[j-1]])
                courbe_12.append([X1[j], Y1[j]])
            
            distx_C2 = abs(X2[j-1]-X1[j])
            disty_C2 = abs(Y2[j-1]-Y1[j])
            if X2[j-1]<X1[j] and distx_C2 <distx_max and distx_C2>distx_min and disty_C2<disty_max:
                courbe_21.append([X2[j-1], Y2[j-1]])
                courbe_22.append([X1[j], Y1[j]])
                
                
    return l1_u, l2_u, courbe_11, courbe_12, courbe_21,courbe_22