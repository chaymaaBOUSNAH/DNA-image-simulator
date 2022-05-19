import numpy as np

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



def Add_Salt(image, amount):
    row,col=  image.shape
    # pourcentage de bruit salt& pepper à ajouter
    # Add Salt 
    # ceil return the smallest integer of a float 
    num_salt = np.ceil(amount * image.size)
    
    # choisir aléatopirement les coordonnées ou on va mettre salt 
    coord = [np.random.randint(0, i-1, int(num_salt)) for i in image.shape]
    
    x, y = coord
    

    #coords = np.array(coord)
    image[(x, y)] = 255
    
    return image