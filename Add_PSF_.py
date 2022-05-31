import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image




def split_channels(im: np.ndarray):
    assert len(im.shape) == 3 and im.shape[-1] == 3
    return np.squeeze(np.split(im, im.shape[-1], -1), axis=-1)

'''
generate a gaussian distribution intensity
gauss_function = a*np.exp(-(x-m)**2/(2*s**2))
'''
def gaus(x,  a, m, s):
    return a*np.exp(-(x-m)**2/(2*s**2))
    # if you want it normalized:
    #return 1/(np.sqrt(2*np.pi*s**2))*np.exp(-(x-m)**2/(2*s**2)) 
    
    
def Add_PSF(image):
    h, w, ch = image.shape
    [R, G, B] = split_channels(image)
    
    a = np.random.randint(10, 40)
    m = np.random.randint(20, 2000)
    s = np.random.randint(5, 50)
    n = np.random.uniform(0.5, 1.2)
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    
    chosen_channel = random.choices(['R', 'G', 'B'], weights=(0.4, 0.5, 0.1))
    print('chosen_channel', chosen_channel)
    if chosen_channel == ['R']:
        R = R + gaus(xx, a*2, m, s)*gaus(yy, a*2, m, s*n)
        G = G + gaus(xx, a, m, s)*gaus(yy, a, m, s*n)
        B = B + gaus(xx, a, m, s)*gaus(yy, a, m, s*n)
    elif chosen_channel == ['G']:
        R = R + gaus(xx, a, m, s)*gaus(yy, a, m, s*n)
        G = G + gaus(xx, a*2, m, s)*gaus(yy, a*2, m, s*n)
        B = B + gaus(xx, a, m, s)*gaus(yy, a, m, s*n)
    else:
        R = R + gaus(xx, a, m, s)*gaus(yy, a, m, s*n)
        G = G + gaus(xx, a, m, s)*gaus(yy, a, m, s*n)
        B = B + gaus(xx, a*2, m, s)*gaus(yy, a*2, m, s*n)
    
    R = np.uint8(np.clip(R, 0, 255))
    G = np.uint8(np.clip(G, 0, 255))
    B = np.uint8(np.clip(B, 0, 255))
   
    return np.dstack((R, G, B))