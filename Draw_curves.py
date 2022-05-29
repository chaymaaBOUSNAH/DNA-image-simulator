import numpy as np
from Bezier_curve import Bezier


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