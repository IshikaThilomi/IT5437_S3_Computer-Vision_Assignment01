#258799R - Manuel I.T
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def manual_histogram_equalization(image):
    # 1. Get image dimensions
    rows, cols = image.shape
    total_pixels = rows * cols
    
    # 2. Calculate PDF (Histogram)
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0,256])
    pdf = hist / total_pixels
    
    # 3. Calculate CDF
    cdf = pdf.cumsum()
    
    # 4. Create lookup table (Normalized CDF * 255)
    lookup_table = (cdf * 255).astype(np.uint8)
    
    # 5. Map original pixels to new values
    equalized_image = lookup_table[image]
    
    return equalized_image, hist, lookup_table

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, '..', 'Images', 'runway.png')
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is not None:
    # Apply custom function
    res_img, original_hist, mapping = manual_histogram_equalization(img)
    
    # Visualization for Report 
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1); plt.imshow(img, cmap='gray'); plt.title('Original Runway')
    plt.subplot(2, 2, 2); plt.imshow(res_img, cmap='gray'); plt.title('Equalized Runway')
    
    plt.subplot(2, 2, 3); plt.plot(original_hist); plt.title('Original Histogram')
    plt.subplot(2, 2, 4); plt.hist(res_img.flatten(), bins=256, range=[0,256]); plt.title('Equalized Histogram')
    
    plt.tight_layout()
    
    # Save results
    save_path = os.path.join(script_dir, '..', 'Images', 'Results', 'q3_equalization_results.png')
    plt.savefig(save_path)
    plt.show()