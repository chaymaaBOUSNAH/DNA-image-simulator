import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random
from Add_gaussian_2D_image import Gaussian_noise

def Add_Electronic_noise(image, amount_SP, Gaussian_noise_sigma, gaussian_Blur_sigma):
   
    row,col,ch= image.shape
    
    output = image.copy()
    
    blue_channel = output[:, :, 2]
    red_channel = output[:, :, 0]
    green_channel = output[:, :, 1]
    
    channels = [red_channel, green_channel, blue_channel]
    noisy_image = []
    for i in range(ch):
        channel = output[: , :, i]
        # le poucentage de salt vs pepper
        s_vs_p = np.random.uniform(0.7, 0.9) 
        # pourcentage de bruit salt& pepper à ajouter
        amount = np.random.uniform(0.1, amount_SP)
        print('amount salt fpr channel ', i, 'est', amount)
        # Add Salt 
        # ceil return the smallest integer of a float 
        num_salt = np.ceil(amount * channel.size * s_vs_p)
        
        # choisir aléatopirement les coordonnées ou on va mettre salt 
        coord = [np.random.randint(0, i-1, int(num_salt)) for i in channel.shape]
        
        x, y = coord
        

        #coords = np.array(coord)
        channel[(x, y)] = np.array([1], dtype='uint8')
        #channel[(random_xcoord, random_ycoord)] = np.array([1], dtype='uint8')
          
        # Add Pepper 
        num_pepper = np.ceil(amount* channel.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in channel.shape]
        
        _x, _y = coords
 
        #coords = np.array(coords)
        channel[(_x, _y)] = np.array([0], dtype='uint8')
        #channel[(_xcoord, _ycoord)] = np.array([0], dtype='uint8')
        
        # Add gaussian noise to each channel
        #channel_gauss = Gaussian_noise(channel, Gaussian_noise_sigma)
        # blur each channel
        channel_blur = gaussian_filter(channel, sigma=gaussian_Blur_sigma)
        
        noisy_image.append(channel_blur)
        
        '''
        # enlever les valeurs négatives et les valeurs >255
        for i in range(channel.shape[0]):
            for j in range(channel.shape[1]):
                if channel[i][j]<0:
                    channel[i][j] = 0
                elif channel[i][j]>255:
                    channel[i][j] = 255
        '''
    noisy_image = np.array(noisy_image)
    noisy = noisy_image.transpose(1, 2, 0)
    #plt.imshow(noisy)
    return noisy


from PIL import Image, ImageFilter
amount_SP = 0.4
Gaussian_noise_sigma = 0.5
gaussian_Blur_sigma = 2

image_path = './noisy_fibers/image_0.png'
image = plt.imread(image_path)
image = image[:, :, :3]


noisy = Add_Electronic_noise(image, amount_SP, Gaussian_noise_sigma, gaussian_Blur_sigma)
noisy = (noisy*255).astype(np.uint8)
radius = np.random.randint(2, 6)  
print('radius', radius)
pil_image=Image.fromarray(noisy)
pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius = radius))

pil_image.show()