import numpy as np
from Bezier_curve import Bezier
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




# dessiner du bruit en forme de u entre deux fibre 
# choisit selon des condition lors de la création initiale des images

def draw_cercle(P1, P2):
    x1, y1 = P1[1], P1[0]
    x2, y2 = P2[1], P2[0]
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