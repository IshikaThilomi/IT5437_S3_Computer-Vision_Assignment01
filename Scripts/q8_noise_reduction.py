# 258799R - Manuel I.T
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # 1. Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, '..', 'Images', 'rice.png') 
    results_dir = os.path.join(script_dir, '..', 'Images', 'Results')
    os.makedirs(results_dir, exist_ok=True)
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found.")
        return

    # --- (a) Apply Gaussian Smoothing ---
    # We use a 5x5 kernel. Notice how this "smears" the noise.
    gaussian_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # --- (b) Apply Median Filtering ---
    # This is the correct choice for salt and pepper noise.
    median_blur = cv2.medianBlur(img, 5)

    # --- Additional Step: Thresholding for comparison ---
    # To show why Median is better, we threshold the median-filtered version
    _, binary = cv2.threshold(median_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Visualization for Report
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(img, cmap='gray'); plt.title('Original (Noise)')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(gaussian_blur, cmap='gray'); plt.title('(a) Gaussian Smoothing')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(median_blur, cmap='gray'); plt.title('(b) Median Filtering')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(binary, cmap='gray'); plt.title('Final Thresholded')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the comparison for your report
    save_path = os.path.join(results_dir, 'q8_filtering_comparison.png')
    plt.savefig(save_path)
    print(f"Comparison results saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()