import re
import math

def sorted_file( l ): 
    
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def distance(P1, P2):
    dist = math.sqrt((P1[0]-P2[0])**2+(P1[1]-P2[1])**2)
    return dist