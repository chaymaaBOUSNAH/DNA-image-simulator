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