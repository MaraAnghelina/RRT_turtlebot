import cv2
import numpy as np 
import matplotlib.pyplot as plt
import yaml

def plot_obstacle_poly(ax, color, poly, img):
    #upload the YAML file
    yamlFile = "map.yaml"
    with open(yamlFile, 'r') as f:
        map_metadata = yaml.safe_load(f)

    resolution = map_metadata["resolution"]
    origin = map_metadata["origin"]

    #upload image in PGM
    map_image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    height, width = map_image.shape

    map_width = width * resolution  
    map_height = height * resolution

    _, binary_map = cv2.threshold(map_image, 200, 255, cv2.THRESH_BINARY_INV)

    #create an obstacle map where 0 is an obstacle and 1 is free space
    obstacle_map = np.where(binary_map == 0, 1, 0)

    ax.imshow(obstacle_map, cmap='gray', origin='upper', extent=[origin[0], origin[0] + map_image.shape[1] * resolution, 
                                                                 origin[1], origin[1] + map_image.shape[0] * resolution])


    ax.set_xlim(origin[0], origin[0] + map_height)
    ax.set_ylim(origin[1], origin[1] + map_width)

    obstacle_pixels = np.where(map_image == 0) 

    obstacle_coords = []
    for row, col in zip(*obstacle_pixels):
    
        x_global = origin[0] + col * resolution
        y_global = origin[1] + (height - row - 1) * resolution  # in Y, inversam ordinea pentru că OpenCV folosește o coordonată Y inversa

        obstacle_coords.append((x_global, y_global))

    poly = obstacle_coords
    x_coords = [coord[0] for coord in obstacle_coords]
    y_coords = [coord[1] for coord in obstacle_coords]

    # ploteaza folosind ax.plot(), pentru puncte
    #ax.plot(x_coords, y_coords, 'ko', markersize=5) 
    

    return poly



