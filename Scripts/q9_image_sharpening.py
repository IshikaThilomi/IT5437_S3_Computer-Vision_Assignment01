# 258799R - Manuel I.T
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def manual_bilateral_filter(image, d, sigma_s, sigma_r):
    """
    Manually implements a bilateral filter for grayscale images.
    """
    img = image.astype(np.float32)
    rows, cols = img.shape
    half_d = d // 2
    output = np.zeros_like(img)

    # 1. Precompute Spatial Gaussian component
    x, y = np.meshgrid(np.arange(d) - half_d, np.arange(d) - half_d)
    spatial_gaussian = np.exp(-(x**2 + y**2) / (2 * sigma_s**2))

    # 2. Iterate through pixels (ignoring borders for simplicity)
    for i in range(half_d, rows - half_d):
        for j in range(half_d, cols - half_d):
            # Extract local neighborhood
            neighborhood = img[i - half_d : i + half_d + 1, j - half_d : j + half_d + 1]
            
            # Calculate Range (intensity) Gaussian component
            intensity_diff = neighborhood - img[i, j]
            range_gaussian = np.exp(-(intensity_diff**2) / (2 * sigma_r**2))
            
            # Combined Weight
            weights = spatial_gaussian * range_gaussian
            
            # Normalize and apply
            output[i, j] = np.sum(weights * neighborhood) / np.sum(weights)

    return output.astype(np.uint8)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, '..', 'Images', 'rice.png')
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None: return

    # (d) Manual Implementation [cite: 62]
    # Parameters: diameter=5, sigma_s=25, sigma_r=25
    manual_res = manual_bilateral_filter(img, 5, 25.0, 25.0)

    # (c) OpenCV Implementation [cite: 61]
    opencv_res = cv2.bilateralFilter(img, 5, 25, 25)

    # Visualization
    plt.figure(figsize=(15, 5))
    plt.subplot(131); plt.imshow(img, cmap='gray'); plt.title('Original')
    plt.subplot(132); plt.imshow(manual_res, cmap='gray'); plt.title('Manual Bilateral')
    plt.subplot(133); plt.imshow(opencv_res, cmap='gray'); plt.title('OpenCV Bilateral')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()