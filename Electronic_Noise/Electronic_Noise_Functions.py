from PIL import Image
import numpy as np
from os import walk
from os.path import join
import random
from scipy.ndimage import gaussian_filter

def split_channels(im: np.ndarray):
    assert len(im.shape) == 3 and im.shape[-1] == 3, 'Verify image shape'
    return np.squeeze(np.split(im, im.shape[-1], -1), axis=-1)



def Gaussian_noise(image, sigma):
   row,col=  image.shape
   mean = 0
   "Draw random samples from a normal (Gaussian) distribution."
   # row * col * ch samples are drawn
   gauss = np.random.normal(mean,sigma,(row,col)) 
   gauss = gauss.reshape(row,col)
   noisy = image + gauss
   return noisy

def Gaussian_noise_RGB(image, sigma):

   row,col,ch= image.shape
   mean = 0# Mean (“centre”) of the distribution.

   "Draw random samples from a normal (Gaussian) distribution."
   # row * col * ch samples are drawn
   gauss = np.random.normal(mean,sigma,(row,col,ch)) 
   gauss = gauss.reshape(row,col,ch)
   noisy = image + gauss
   return noisy

'''
generating a gradient image
'''
def get_gradient_2d(start, stop, width, height):
    is_horizontal = random.choices(['true', 'false'], weights=[0.5, 0.5])
    if is_horizontal==['true']:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T
    
def get_gradient_3d(width, height, start_list, stop_list):
    result = np.zeros((height, width, len(start_list)), dtype=np.float64)

    for i, (start, stop) in enumerate(zip(start_list, stop_list)):
        result[:, :, i] = get_gradient_2d(start, stop, width, height)

    return result

'''
Ajouter du bruit impulsive : grains de sel sur une image 2D
'''
def Add_Salt(image, amount, noise_value, size):
    row,col=  image.shape
    # size la taille en pixel si 4 alors squre == 4X4
    N_salt = np.ceil(amount*image.size)
    # choisir aléatopirement les coordonnées à partir de l'image ou on va mettre salt 
    coord = [np.random.randint(0, i-1, int(N_salt)) for i in (row-size, col-size)]
    x, y = coord
    
    for k in range(len(x)):
        for i in range(size):
            for j in range(size):
                pixels_coord = (x[k]+i, y[k]+j)  
              
                image[pixels_coord] = noise_value
    return image

'''
Ajouter les différents type de bruit sur chaque channal separemment: Salt, Gaussian noise, Blur
'''
def Add_channel_noise(channel, channel_index, dominant_channel, amount_SP, noise_value, s_noise, sigma_Gaussian_noise, gaussian_Blur_sigma, Parasites_ch):
    assert channel_index in [0, 1, 2], 'Enter a valid channel index: R=0, G=1, B=2'
    row,col = channel.shape
    if dominant_channel==['green'] and channel_index != 1:
        amount = amount_SP*0.01
    elif dominant_channel==['red'] and channel_index != 0:
        amount = amount_SP*0.1
    else:
        amount = amount_SP
   
    #Ajouter du bruit: parasites de photons
    channel = channel + Parasites_ch*np.ones((row,col))
    # Ajouter des grainds de sel de différentes tailles sur chaque channal
    channel = Add_Salt(channel, amount, noise_value=noise_value, size=s_noise)
    # Blur
    channel = gaussian_filter(channel, sigma=gaussian_Blur_sigma)
    # Ajouter du bruit gaussien
    channel =  Gaussian_noise(channel, sigma_Gaussian_noise)    
    channel = gaussian_filter(channel, sigma=gaussian_Blur_sigma)
    channel = np.uint8(np.clip(channel, 0, 255))

    return channel


'''
generate a gaussian distribution intensity
gauss_function = a*np.exp(-(x-m)**2/(2*s**2))
'''
def gaus(x,  a, m, s):
    return a*np.exp(-(x-m)**2/(2*s**2))
    # if you want it normalized:
    #return 1/(np.sqrt(2*np.pi*s**2))*np.exp(-(x-m)**2/(2*s**2)) 
    
def Add_PSF_to_channel(channel, m1, m2, a1, s1, a2, s2):
    h, w = channel.shape

    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    

    channel = channel + gaus(xx, a1, m1, s1)*gaus(yy, a2, m2, s2)
            
    channel = np.uint8(np.clip(channel, 0, 255))

   
    return channel    

def Add_PSF_to_image(image, channel_psf, m1, m2, a1, s1, a2, s2):
    n = np.random.randint(2, 4)
    for i in range(image.shape[2]):
        # augmenté l'intensité pour le channal dominanat
        if i == channel_psf[0]:
            k1 = a1*n
            k2 = a2*n
        else:
            k1 = a1
            k2 = a2
    
        image[:, :, i] = Add_PSF_to_channel(image[:, :, i], m1, m2, k1, s1, k2, s2)

    return image   

    
'''
Dégrader les fibres et les analogues sur chaque channal en ajoutant du bruit gaussian 
et en remplassant les valeur inférieur d'une valeurs données par 0 : appliquer mask <255 ==0'
'''    
def degraded_fibers(image, sigma):
    
    output = image.copy()

    row,col,ch= output.shape
    red_channel = np.uint8(np.clip(Gaussian_noise(output[:, :, 0], sigma), 0, 255))
    green_channel = np.uint8(np.clip(Gaussian_noise(output[:, :, 1], sigma), 0, 255))
    blue_channel = np.uint8(np.clip(Gaussian_noise(output[:, :, 2], sigma), 0, 255))
  
    black = np.array([0], dtype='uint8')
    
    red_channel[red_channel <255] = black 
    # or use np.where(red_channel <200, 0, red_channel)
    green_channel[green_channel <255] = black
    blue_channel[blue_channel <255] = black
    
    result_image = np.dstack((red_channel, green_channel, blue_channel))
        
    return result_image

