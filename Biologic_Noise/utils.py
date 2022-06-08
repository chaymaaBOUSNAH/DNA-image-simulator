import math
import numpy as np

def canvas2rgb_array(canvas):
    
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    ncols, nrows = canvas.get_width_height()

    scale = round(math.sqrt(buf.size / 3 / nrows / ncols))
    return buf.reshape(scale * nrows, scale * ncols, 3)

def distance(P1, P2):
    dist = math.sqrt((P1[0]-P2[0])**2+(P1[1]-P2[1])**2)
    return dist