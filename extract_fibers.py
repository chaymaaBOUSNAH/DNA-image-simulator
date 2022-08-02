import matplotlib.pyplot as plt 
import numpy as np
import math  
import pandas as pd
import cv2
from extract_coord import *
#import isect_segments_bentley_ottmann.poly_point_isect as bot
# step_1: détecter les lignes droite sur le canal bleu et extraire les coordonnées de chaque fibre
# step_2: éliminer les ligne de longueur -50 et enregister le reste dans un fichier csv
# step_3: extraire les coordonnées des analogues à partir des chanaux rouge et vert
# step_4: Chercher les fibres qui contienent des analogues = coordonnées des analogues est dans l'intervales des coordonnées des fibres enregistrées
# ou chercher la supperposistion des fibres avec les analogues
# step_5: enregister dans un fichier csv les coordonnées de ces fibres et des analogues, ajouter aussi leurs longueur et combien d'analogues contient chaque fibre et leurs types



def extract_fibers_coordonates(segmented_img):
    all_results = []  # <-- list for all results

    N_fiber = 0
    N_analog = 0
         
    img = segmented_img[:, :, :3]
    thresholds = [150, 20, 20]
    line_lengths = [50, 20, 20]
    line_gaps = [100, 20, 20]
    
    fibers_analogs_coords = []
    image = []
    
    for i , (thresh, line_length, line_gap) in enumerate(zip(thresholds, line_lengths, line_gaps)):
        #print(i)
        channel = img[:, :, i]
        #threshold=100, # Min number of votes for valid line (intersections in Hough grid cell)
        #line_length: minimum number of pixels making up a line
        #rho distance resolution in pixels of the Hough grid
        # theta angular resolution in radians of the Hough grid
        edges = cv2.Canny(channel, 100, 200)
        lines = cv2.HoughLinesP(edges, 1, np.pi/360, threshold = thresh, minLineLength=line_length, maxLineGap=line_gap)
        #print(lines)
        
        if lines is not None:
            # prepare
            _lines = []
            for _line in get_lines(lines):
                _lines.append([(_line[0], _line[1]),(_line[2], _line[3])])

            # sort
            _lines_x = []
            _lines_y = []
            
            for line_i in _lines:
                orientation_i = np.arctan2((line_i[0][1]-line_i[1][1]),(line_i[0][0]-line_i[1][0]))
                if (abs(np.degrees(orientation_i)) > 45) and abs(np.degrees(orientation_i)) < (90+45):
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
            #print("process groups lines", len(_lines), len(merged_lines_all))
            img_merged_lines =  np.zeros(channel.shape)
            for line in merged_lines_all:
                cv2.line(img_merged_lines, (line[0][0], line[0][1]), (line[1][0],line[1][1]), 255, 2)
            
            
            image.append(img_merged_lines)
            
            
        else:
            merged_lines_all = None
            
        fibers_analogs_coords.append(merged_lines_all)
        
    
    # Enregister les coordonnées des fibres et des analogues dans un fichier csv
    fibers = pd.DataFrame(fibers_analogs_coords[2], dtype='f', columns=['P1', 'P2'])
    Analog1 = pd.DataFrame(fibers_analogs_coords[1], dtype='f', columns=['P1', 'P2'])
    Analog2 = pd.DataFrame(fibers_analogs_coords[0], dtype = 'f',columns=['P1', 'P2'])
    '''
    #result = [fibers, Ldu, cldu]
    header = ['fibers', 'Edu', 'cldu']
    result = pd.concat([fibers, Analog1, Analog2], keys=header)
    
    
    result = {'fibers': [fibers_analogs_coords[0]], 'Edu': [fibers_analogs_coords[1]], 'cldu': [fibers_analogs_coords[2]]}
    
    
    
    
    print('result', result_dataframe)
    #result = pd.DataFrame(result, columns=header,  dtype='f')
    '''
            
    return fibers, Analog1, Analog2
            
'''
import PIL      
im = PIL.Image.open('pred_out/BSQ_B140_17.tif')     
im = np.array(im)
fibers, Analog1, Analog2 = extract_fibers_coordonates(np.array(im))
print(fibers)              
'''