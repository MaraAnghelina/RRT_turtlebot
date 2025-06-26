import cv2
import numpy as np
import yaml

def process_map(image_path, yaml_path, save_path):
    
    with open(yaml_path, 'r') as f:
        map_metadata = yaml.safe_load(f)
    
    resolution = map_metadata["resolution"]
    origin = map_metadata["origin"]

    map_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    _, binary_map = cv2.threshold(map_image, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(255 - binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    xOffset = 0
    yOffset = 0

    if contours:
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        xOffset = origin[0] + x * resolution
        yOffset = origin[1] + y * resolution
        cropped_map = binary_map[y:y+h, x:x+w]  # decupeaza doar zona conturului
    else:
        cropped_map = binary_map  # nu exista contur

    cv2.imwrite(save_path, cropped_map)

    print(f"Imagine procesată salvată ca: {save_path}")
    print("X offset", x)
    print("Y offset", y)

    return save_path, xOffset, yOffset




