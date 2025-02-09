import cv2
import numpy as np 
import matplotlib.pyplot as plt
import yaml

def plot_obstacle_poly(ax, color, poly):
    #Upload the YAML file
    with open("mapRRT.yaml", 'r') as f:
        map_metadata = yaml.safe_load(f)

    resolution = map_metadata["resolution"]
    origin = map_metadata["origin"]

    #upload image in PGM
    map_image = cv2.imread("mapRRT.pgm", cv2.IMREAD_GRAYSCALE)

    height, width = map_image.shape

    map_width = width * resolution  
    map_height = height * resolution


    ax.set_xlim(origin[0], origin[0] + map_height)
    ax.set_ylim(origin[1], origin[1] + map_width)

    obstacle_pixels = np.where(map_image == 0) 

    obstacle_coords = []
    for row, col in zip(*obstacle_pixels):
    
        x_global = origin[0] + col * resolution
        y_global = origin[1] + (height - row - 1) * resolution  # În Y, inversăm ordinea pentru că OpenCV folosește o coordonată Y inversă

        obstacle_coords.append((x_global, y_global))

    poly = obstacle_coords
    x_coords = [coord[0] for coord in obstacle_coords]
    y_coords = [coord[1] for coord in obstacle_coords]

    # Plotază folosind ax.plot(), pentru puncte
    ax.plot(x_coords, y_coords, 'ko', markersize=5) 
    

    return



