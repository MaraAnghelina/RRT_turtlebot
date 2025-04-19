import cv2
import numpy as np
import yaml

def loadYaml(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def findBlackRegion(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    _, binary = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY_INV)
    
    coords = np.column_stack(np.where(binary > 0))
    
    ymin, xmin = coords.min(axis=0)
    ymax, xmax = coords.max(axis=0)
    
    return xmin, xmax, ymin, ymax, img.shape[0]

def pixelToCoord(xmin, xmax, ymin, ymax, yaml_data, img_height):
    resolution = yaml_data['resolution']
    origin_x, origin_y, _ = yaml_data['origin']
    
    x_min_real = origin_x + xmin * resolution
    x_max_real = origin_x + xmax * resolution
    y_min_real = origin_y + (img_height - ymax) * resolution
    y_max_real = origin_y + (img_height - ymin) * resolution
    
    return x_min_real, x_max_real, y_min_real, y_max_real

def intervalCoord(img_path, yaml_path):
    yaml_data = loadYaml(yaml_path)
    xmin, xmax, ymin, ymax, height = findBlackRegion(img_path)
    x_min_real, x_max_real, y_min_real, y_max_real = pixelToCoord(xmin, xmax, ymin, ymax, yaml_data, height)
    
    #print(f"Pixel coord: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")
    #print(f"Real coord: x_min={x_min_real}, x_max={x_max_real}, y_min={y_min_real}, y_max={y_max_real}")

    return x_min_real, x_max_real, y_min_real, y_max_real

    
    
