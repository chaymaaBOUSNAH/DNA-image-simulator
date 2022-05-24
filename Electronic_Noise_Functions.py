from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from os import walk
from os.path import join
import random



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

   sigma = np.random.uniform(0.2, 0.7)
   "Draw random samples from a normal (Gaussian) distribution."
   # row * col * ch samples are drawn
   gauss = np.random.normal(mean,sigma,(row,col,ch)) 
   gauss = gauss.reshape(row,col,ch)
   noisy = image + gauss
   return noisy



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
   
        

    '''
    image[(x, y)] = noise_value
    image[(x+1, y)] = noise_value
    image[(x, y+1)] = noise_value
    image[(x+1, y+1)] = noise_value
    '''
    return image


def degraded_fibers(image, sigma):
    
    output = image.copy()

    row,col,ch= output.shape
    
    output_gaussian = Gaussian_noise_RGB(output, sigma)
    output_gaussian  = np.uint8(np.clip(output_gaussian, 0, 255))
    red_channel = output_gaussian[:, :, 0]
    green_channel = output_gaussian[:, :, 1]
    blue_channel = output_gaussian[:, :, 2]
  
    black = np.array([0], dtype='uint8')
    
    red_channel[red_channel <255] = black 
    # or use np.where(red_channel <200, 0, red_channel)
    green_channel[green_channel <255] = black
    blue_channel[blue_channel <255] = black
    
    result_image = np.dstack((red_channel, green_channel, blue_channel))
        
    return result_image


def Add_glow(image, glue_dir, prob):
    row,col,ch= image.shape
    imag = Image.fromarray(np.uint8(np.clip(image, 0, 255)))
    Imagecopy = imag.copy() 
    
    dir_path = glue_dir
    for (dirpath, dirnames, file_names) in walk(dir_path):
        # n est le nombre aléatoire d'image de glue qui seront ajouter
        add_glue = ['true', 'false']
        adding_glue = random.choices(add_glue, weights=[prob, 1-prob])
        if adding_glue==['true']:
            n = np.random.randint(1, 3)
            random_glue = random.sample(file_names, n)
            for glue in random_glue:
                
                glue_h = np.random.randint(50, 150) 
                glue_w = np.random.randint(50, 150) 
                glue_path = join(dir_path, glue)
                glue_img = Image.open(glue_path)
                
                #angle = random.choice([0, 90, 180])
                #glue_img = glue_img.rotate(angle)
                gluecopy = glue_img.copy() 
                
                a = np.random.randint(0, row-glue_h)
                b = np.random.randint(0, col-glue_w)
                
                Imagecopy.paste(gluecopy.resize((glue_h, glue_w)), (a, b))
    return np.array(Imagecopy) 