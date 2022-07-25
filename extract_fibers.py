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
from extract_coord import *
#import isect_segments_bentley_ottmann.poly_point_isect as bot
# step_1: détecter les lignes droite sur le canal bleu et extraire les coordonnées de chaque fibre
# step_2: éliminer les ligne de longueur -50 et enregister le reste dans un fichier csv
# step_3: extraire les coordonnées des analogues à partir des chanaux rouge et vert
# step_4: Chercher les fibres qui contienent des analogues = coordonnées des analogues est dans l'intervales des coordonnées des fibres enregistrées
# ou chercher la supperposistion des fibres avec les analogues
# step_5: enregister dans un fichier csv les coordonnées de ces fibres et des analogues, ajouter aussi leurs longueur et combien d'analogues contient chaque fibre et leurs types


all_results = []  # <-- list for all results

Segmented_images_path = './mesures/predictions/'

for (dirpath, dirnames, filenames) in walk(Segmented_images_path):
    c = 0
    for image_file in filenames:
        print(image_file)
        N_fiber = 0
        N_analog = 0
             
        image_path = join(Segmented_images_path, image_file)
        img = cv2.imread(image_path)
        img = img[:, :, :3]
        thresholds = [50, 10, 10]
        line_lengths = [30, 10, 10]
        line_gaps = [30, 10, 10]
        
        fibers_analogs_coords = []
        image = []
        
        for i , (thresh, line_length, line_gap) in enumerate(zip(thresholds, line_lengths, line_gaps)):
            print(i)
            channel = img[:, :, i]
            #threshold=100, # Min number of votes for valid line (intersections in Hough grid cell)
            #line_length: minimum number of pixels making up a line
            #rho distance resolution in pixels of the Hough grid
            # theta angular resolution in radians of the Hough grid
            edges = cv2.Canny(channel, 10, 100)
            lines = cv2.HoughLinesP(edges, 1, np.pi/360, threshold = thresh, minLineLength=line_length, maxLineGap=line_gap)
            print(lines)
            
            if lines is not None:
                # prepare
                _lines = []
                for _line in get_lines(lines):
                    _lines.append([(_line[0], _line[1]),(_line[2], _line[3])])

                # sort
                _lines_x = []
                _lines_y = []
                
                for line_i in _lines:
                    orientation_i = math.atan2((line_i[0][1]-line_i[1][1]),(line_i[0][0]-line_i[1][0]))
                    if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90+45):
                        _lines_y.append(line_i)
                    else:
                        _lines_x.append(line_i)
    
                _lines_x = sorted(_lines_x, key=lambda _line: _line[0][0])
                _lines_y = sorted(_lines_y, key=lambda _line: _line[0][1])
    
                merged_lines_x = merge_lines_pipeline_2(_lines_x)
                merged_lines_y = merge_lines_pipeline_2(_lines_y)
    
                merged_lines_all = []
                merged_lines_all.append(merged_lines_x)
                merged_lines_all.append(merged_lines_y)
                merged_lines_all = merged_lines_all[0]
                print("process groups lines", len(_lines), len(merged_lines_all))
                img_merged_lines =  np.zeros(channel.shape)
                for line in merged_lines_all:
                    cv2.line(img_merged_lines, (line[0][0], line[0][1]), (line[1][0],line[1][1]), 255, 2)
                plt.imshow(img_merged_lines)
                plt.show()
                
                image.append(img_merged_lines)
                
                
            else:
                merged_lines_all = None
                
            fibers_analogs_coords.append(merged_lines_all)
            
        
        # Enregister les coordonnées des fibres et des analogues dans un fichier csv
        fibers = pd.DataFrame(fibers_analogs_coords[0], dtype='f')
        Analog1 = pd.DataFrame(fibers_analogs_coords[1], dtype='f')
        Analog2 = pd.DataFrame(fibers_analogs_coords[2], dtype = 'f')
        
        #result = [fibers, Ldu, cldu]
        header = ['fibers', 'Edu', 'cldu']
        result = pd.concat([fibers, Analog1, Analog2], keys=header)
        print('result', result)
        #result = pd.DataFrame(result, columns=header,  dtype='f')
        
        #image = np.array(image).transpose((1, 2, 0))
        #plt.imshow(image)
        #plt.show()
        
        result.to_csv('./mesures/output//output'+str(c)+'.csv', sep=' ', header=True)
        c +=1 
        
        
                    
 