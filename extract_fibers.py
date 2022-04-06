from PIL import Image 
import matplotlib.pyplot as plt 
import numpy as np
import math  
import tifffile
import random
import pandas as pd
import os
import cv2
from os import walk
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from skimage.filters import try_all_threshold, threshold_mean, threshold_otsu, threshold_triangle, threshold_yen
from skimage.transform import hough_circle, hough_circle_peaks, hough_line, hough_line_peaks, probabilistic_hough_line
import csv
from skimage import feature
from bentley_ottmann.planar import segments_intersect
from ground.base import get_context
#import isect_segments_bentley_ottmann.poly_point_isect as bot
# step_1: détecter les lignes droite sur le canal bleu et extraire les coordonnées de chaque fibre
# step_2: éliminer les ligne de longueur -50 et enregister le reste dans un fichier csv
# step_3: extraire les coordonnées des analogues à partir des chanaux rouge et vert
# step_4: Chercher les fibres qui contienent des analogues = coordonnées des analogues est dans l'intervales des coordonnées des fibres enregistrées
# ou chercher la supperposistion des fibres avec les analogues
# step_5: enregister dans un fichier csv les coordonnées de ces fibres et des analogues, ajouter aussi leurs longueur et combien d'analogues contient chaque fibre et leurs types


all_results = []  # <-- list for all results

Segmented_images_path = './sm_with_coord/essai/IA/output/segmented_dir'
for (dirpath, dirnames, filenames) in walk(Segmented_images_path):
    c = 0
    for image_file in filenames:
        N_fiber = 0
        N_analog = 0
             
        image_path = join(Segmented_images_path, image_file)
        img = cv2.imread(image_path)
        img = img[:, :, :3]
        Bleu_channel = img[:, :, 0]
        Green_channel = img[:, :, 1]
        Red_channel = img[:, :, 2]
        

       
        # load image with Pillow as RGB
        img = Image.open(image_path)
        img = np.asarray(img)
        img = img[:, :, :3]
        Red_channel = img[:, :, 0]
        Green_channel = img[:, :, 1]
        Bleu_channel = img[:, :, 2]
        
        #threshold=100, # Min number of votes for valid line (intersections in Hough grid cell)
        #line_length: minimum number of pixels making up a line
        #rho distance resolution in pixels of the Hough grid
        # theta angular resolution in radians of the Hough grid
        edges = feature.canny(Bleu_channel, sigma=3)
        fibers = probabilistic_hough_line(edges, threshold=200, line_length=100, line_gap=10)
        
        # extraire les analogues sur chaque fibre ectraites à partir des canux rouge et vert

        Edu = probabilistic_hough_line(Red_channel, threshold=10, line_length=20, line_gap=5)
        cldu = probabilistic_hough_line(Green_channel, threshold=10, line_length=20, line_gap=5)
        
        
        fig, ax = plt.subplots(3, 2, figsize=(15, 15), sharex=True, sharey=True)
        #ax = axes.ravel()
        
        ax[0, 0].imshow(edges, cmap=plt.cm.gray)
        ax[0, 0].set_title('Input image')
        
        X1 = []
        X2 = []
        Y1 = []
        Y2 = []
        for line in fibers:
            p0, p1 = line
            X1.append(p0[0])
            Y1.append(p0[1])
            X2.append(p1[0])
            Y2.append(p1[1])
            
            ax[0, 1].plot((p0[0], p1[0]), (p0[1], p1[1]))
        ax[0, 1].set_xlim((0, Bleu_channel.shape[1]))
        ax[0, 1].set_ylim((Bleu_channel.shape[0], 0))
        ax[0, 1].set_title('Probabilistic Hough')   
        
        
        ax[1, 0].imshow(Red_channel, cmap=plt.cm.gray)
        ax[1, 0].set_title('Input image')
        for line in Edu:
            p0, p1 = line
            ax[1, 1].plot((p0[0], p1[0]), (p0[1], p1[1]))
        ax[1, 1].set_xlim((0, Red_channel.shape[1]))
        ax[1, 1].set_ylim((Red_channel.shape[0], 0))
        ax[1, 1].set_title('Probabilistic Hough')
        
        
        ax[2, 0].imshow(Green_channel, cmap=plt.cm.gray)
        ax[2, 0].set_title('Input image')
        for line in cldu:
            p0, p1 = line
            ax[2, 1].plot((p0[0], p1[0]), (p0[1], p1[1]))
        ax[2, 1].set_xlim((0, Green_channel.shape[1]))
        ax[2, 1].set_ylim((Green_channel.shape[0], 0))
        ax[2, 1].set_title('Probabilistic Hough')
        
        fig.show()
        
        '''
        context = get_context()
        Point, Segment = context.point_cls, context.segment_cls
        
        
        for i in range(len(fibers)):
            unit_segments = [Segment(Point(X1[i], Y1[i]), Point(X2[i], Y2[i])), Segment(Point(0, 0), Point(0, 1))]
        segments = Segment
        intersection = segments_intersect(points)
        print('inetersection', intersection)
        '''
        fibers = pd.DataFrame(fibers, dtype='f')
        Analog1 = pd.DataFrame(Edu, dtype='f')
        Analog2 = pd.DataFrame(cldu, dtype = 'f')
        
        #result = [fibers, Ldu, cldu]
        header = ['fibers', 'Edu', 'cldu']
        result = pd.concat([fibers, Analog1, Analog2], keys=header)
        print('result', result)
        #result = pd.DataFrame(result, columns=header,  dtype='f')
       
        result.to_csv('./sm_with_coord/essai/IA/output/output'+str(c)+'.csv', sep=' ', header=True)
        c +=1 
        
"""
 
        all_results.append(result)  # <-- keep result
         
                
                
        fh = open('./sm_with_coord/essai/IA/output/output'+str(c)+'.csv', 'w', newline='')
        csv_writer = csv.writer(fh) 
        csv_writer.writerow(header)
        csv_writer.writerows(all_results)
                                                                                                
        fh.close()
        c +=1         
                    
 
        
       
        
        
        # This returns an array of r and theta values
        lines = cv2.HoughLines(Bleu_channel,1,np.pi/180, 500)
   
        if lines is not None:
            print(lines)
            # The below for loop runs till r and theta values
            # are in the range of the 2d array
            Line = []
            for r_theta in lines:
                print('theta', r_theta)
                r,theta = r_theta[0]
                # Stores the value of cos(theta) in a
                a = np.cos(theta)
             
                # Stores the value of sin(theta) in b
                b = np.sin(theta)
                 
                # x0 stores the value rcos(theta)
                x0 = a*r
                 
                # y0 stores the value rsin(theta)
                y0 = b*r
                 
                # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
                x1 = int(x0 + 1000*(-b))
                 
                # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
                y1 = int(y0 + 1000*(a))
             
                # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
                x2 = int(x0 - 1000*(-b))
                 
                # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
                y2 = int(y0 - 1000*(a))
                
                p1 = (x1, y1)
                p2 = (x2, y2)
                
                 
                line = (p1, p2)
                Line.append(line)
                # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
                # (0,0,255) denotes the colour of the line to be
                #drawn. In this case, it is red.
                #cv2.line(img,(x1,y1), (x2,y2), (255, 0, 0), 5)
                # Generating figure 2
        
                #cv2.imwrite('linesDetected.jpg', img)
            
        else: 
            print('No line detected')
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(Bleu_channel, cmap=plt.cm.gray)
        ax[0].set_title('Input image')


        ax[1].imshow(Bleu_channel * 0)
        for line in Line:
            p0, p1 = line 
            ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]))
        ax[1].set_xlim((0, Bleu_channel.shape[1]))
        ax[1].set_ylim((Bleu_channel.shape[0], 0))
        ax[1].set_title('Probabilistic Hough')

                 
            # All the changes made in the input image are finally
            # written on a new image houghlines.jpg
            
"""
        
