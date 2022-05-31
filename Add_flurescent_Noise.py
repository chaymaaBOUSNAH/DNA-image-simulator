from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from utils import sorted_file, canvas2rgb_array
from Draw_curves import draw_cercle
import time
'''
fonction qui retourne une figure avec bruit coloré et flurescent
'''
def Add_Flurescent_noise(image, Flurescent_noise, pepper):
    start_time = time.time()
    assert np.amax(image) <= 255 and np.amin(image) >= 0,  'values are not in the range [0 255], normalise image values'
    
    fig, ax = plt.subplots(1, 1, figsize=(2048, 2048), sharex=True, sharey=True, dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    fig.set_size_inches(2048/100, 2048/100)
    ax.margins(0)
    ax.imshow(image, cmap=plt.cm.gray, aspect='auto')

    
    for noisy_dust in range(Flurescent_noise):   
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
   

    ax.set_xlim((0, image.shape[1]))
    ax.set_ylim((image.shape[0], 0))
    
    print('Flurescent noise', "--- %s seconds ---" % (time.time() - start_time))  
     
    return canvas2rgb_array(fig.canvas)

