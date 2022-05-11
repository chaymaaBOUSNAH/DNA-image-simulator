from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from utils import sorted_file, canvas2rgb_array


'''
fonction qui retourne une figure avec bruit coloré et flurescent
'''
def Add_Flurescent_noise(image_path, Flurescent_noise, pepper):

    imag = plt.imread(image_path)
    
    image = imag[:,:, :3]
    
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
        colors = ['#32CD32', '#FF00FF']
        color = random.choices(colors, weights=[0.8, 0.2])
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
    for j in range(pepper):
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
        
        #plt.scatter(x, y, s=size, c=color[0], marker=markr[0])    
    ax.set_xlim((0, image.shape[1]))
    ax.set_ylim((image.shape[0], 0))
     
     
    return fig


Flurescent_noise = np.random.randint(50, 200)
pepper = np.random.randint(6000, 12000)
image_path = './create_images/Essai/1.png'

fig = Add_Flurescent_noise(image_path, Flurescent_noise, pepper)

img = canvas2rgb_array(fig.canvas)
pil_image=Image.fromarray(img)
pil_image.show()