# 258799R - Manuel I.T
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # 1. Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, '..', 'Images', 'runway.png')
    results_dir = os.path.join(script_dir, '..', 'Images', 'Results')
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return

    # 2. Apply Laplacian Filter (Extract Edges)
    # We use CV_64F to avoid overflow during subtraction
    laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    
    # 3. Sharpening: Original - Laplacian
    # Since the center of the Laplacian kernel is positive (4), 
    # subtracting the second derivative enhances the edges.
    sharpened = img.astype(np.float64) - laplacian
    
    # Clip values to [0, 255] and convert back to uint8
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # 4. Visualization for Report
    plt.figure(figsize=(15, 5))
    plt.subplot(131); plt.imshow(img, cmap='gray'); plt.title('Original Runway')
    plt.axis('off')
    
    plt.subplot(132); plt.imshow(np.abs(laplacian), cmap='gray'); plt.title('Laplacian (Edges)')
    plt.axis('off')
    
    plt.subplot(133); plt.imshow(sharpened, cmap='gray'); plt.title('Sharpened Result')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'q9_sharpening_results.png'))
    plt.show()

if __name__ == "__main__":
    main()