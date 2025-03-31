import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_map():
    # Simple direct approach using OpenCV
    print("Reading the map...")
    img = cv2.imread('map.pgm', cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print("Failed to read the map. Trying with -1 flag...")
        img = cv2.imread('map.pgm', -1)
    
    if img is None:
        print("Still failed to read the map. Please check if the file exists.")
        return
    
    print(f"Map loaded successfully. Shape: {img.shape}, Data type: {img.dtype}")
    print(f"Min value: {img.min()}, Max value: {img.max()}")
    
    # Display original image
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title("Original Map")

    kernel = np.ones((5, 5), np.uint8)
    inverted = 255 - img
    _, binary_inv = cv2.threshold(inverted, 128, 255, cv2.THRESH_BINARY)
    processed_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel)
    processed_inv = 255 - processed_inv
    
    cv2.imwrite('processed_map_inv.png', processed_inv)
    cv2.imwrite('processed_map_inv.pgm', processed_inv)
    
    # Load the map
    img = cv2.imread('processed_map_inv.pgm', cv2.IMREAD_UNCHANGED)

    # Invert the map
    inverted = 255 - img

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(inverted, (3, 3), 0)

    # Binary thresholding
    _, binary = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)

    # Morphological operations
    kernel = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Invert back to original convention
    result = 255 - closed

    # Save the processed map
    cv2.imwrite('refined_map_blur.pgm', result)
    cv2.imwrite('refined_map_blur.png', result)
    
    # Display processed image
    plt.subplot(122)
    plt.imshow(processed_inv, cmap='gray')
    plt.title("Processed Map")
    plt.tight_layout()
    plt.savefig('comparison.png')
    plt.show()
    
    

if __name__ == "__main__":
    process_map()