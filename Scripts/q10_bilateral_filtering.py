# 258799R - Manuel I.T
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def manual_bilateral_filter(image, d, sigma_s, sigma_r):
    """
    Manually implements a bilateral filter for grayscale images[cite: 57, 58].
    """
    img = image.astype(np.float32)
    rows, cols = img.shape
    half_d = d // 2
    output = np.zeros_like(img)

    # 1. Precompute Spatial Gaussian component (distance stays constant for all windows)
    x, y = np.meshgrid(np.arange(d) - half_d, np.arange(d) - half_d)
    spatial_gaussian = np.exp(-(x**2 + y**2) / (2 * sigma_s**2))

    # 2. Iterate through pixels (padding handled by ignoring borders)
    for i in range(half_d, rows - half_d):
        for j in range(half_d, cols - half_d):
            # Extract local neighborhood
            neighborhood = img[i - half_d : i + half_d + 1, j - half_d : j + half_d + 1]
            
            # 3. Calculate Range (intensity) Gaussian component
            # This depends on the intensity difference from the center pixel [cite: 58]
            intensity_diff = neighborhood - img[i, j]
            range_gaussian = np.exp(-(intensity_diff**2) / (2 * sigma_r**2))
            
            # 4. Combined Weight (Spatial * Range)
            weights = spatial_gaussian * range_gaussian
            
            # 5. Normalize and apply to center pixel
            output[i, j] = np.sum(weights * neighborhood) / np.sum(weights)

    return output.astype(np.uint8)

def main():
    # Setup paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, '..', 'Images', 'rice.png')
    results_dir = os.path.join(script_dir, '..', 'Images', 'Results')
    os.makedirs(results_dir, exist_ok=True)
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load {img_path}")
        return

    # Parameters as per standard denoising requirements [cite: 58]
    diameter = 5
    s_space = 25.0
    s_range = 25.0

    # (c) Bilateral filtering using OpenCV [cite: 61]
    opencv_res = cv2.bilateralFilter(img, diameter, s_range, s_space)

    # (d) Manually implemented bilateral filter [cite: 62]
    manual_res = manual_bilateral_filter(img, diameter, s_space, s_range)

    # Visualization and Comparison for Report [cite: 96]
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(img, cmap='gray'); plt.title('Original (Rice)')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(manual_res, cmap='gray'); plt.title('Manual Bilateral Filter')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(opencv_res, cmap='gray'); plt.title('OpenCV cv.bilateralFilter()')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save results for the report
    save_path = os.path.join(results_dir, 'q10_bilateral_results.png')
    plt.savefig(save_path)
    print(f"Bilateral results saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()