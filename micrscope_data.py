import cv2
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from os.path import isfile, join
from os import walk

mypath_images = './data/img_nouv'
file_index = 0
for (dirpath, dirnames, filenames) in walk(mypath_images):
    for file in filenames:
        f_path = join(mypath_images, file)
        imag = plt.imread(f_path)
        
        plt.imsave('./data/img_nouv2/'+str(file_index)+'.png', imag)
        file_index += 1
        



"""
mypath_cord = './data/essai_mask'
file_index = 0
for (dirpath, dirnames, filenames) in walk(mypath_cord):
    for file in filenames:
        f_path = join(mypath_cord, file)
        print(file)
        cord = pd.read_csv(f_path, sep='\t')
        
        #extract cordonnates:
        Type = np.array(cord['DNA'])
        l = np.array(cord['l'])
        x1 = np.array(cord['x1'])
        y1 = np.array(cord['y1'])
        x2 = np.array(cord['x2'])
        y2 =  np.array(cord['y2'])
        
        image = np.zeros((2048, 2048, 3))
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        dpi=100 
        fig.set_size_inches(2048/dpi, 2048/dpi)
        
        ax.imshow(image)
        
        fibers = []
        Analog1 = []
        Analog2 = []
                
        for i, Type in enumerate(Type):
            print(i, Type)
            if Type=='DNA':
                ax.plot((x1[i], y1[i]), ( x2[i], y2[i]), 'b', linewidth=5)
            elif Type=='ASTAINED':
                ax.plot((x1[i], y1[i]), ( x2[i], y2[i]), 'aqua', linewidth=5)
            elif Type=='BSTAINED':
                ax.plot((x1[i], y1[i]), ( x2[i], y2[i]), 'magenta', linewidth=5)
                
        
        ax.set_xlim((0, image.shape[1]))
        ax.set_ylim((image.shape[0], 0))
        #ax[1].set_title('draw lines')
        plt.savefig('./data/mask_nouv/'+str(file_index)+'_mask', bbox_inches='tight', pad_inches=0, dpi = 100)
        file_index +=1 
        
        plt.tight_layout()
        plt.show()    
                
      
"""