# 258799R - Manuel I.T 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def zoom_image(image, s, method='nearest'):
    old_h, old_w = image.shape[:2]
    new_h, new_w = int(old_h * s), int(old_w * s)
    
    # Create output image
    zoomed = np.zeros((new_h, new_w), dtype=np.uint8)
    
    # Pre-calculate ratios
    row_ratio = old_h / new_h
    col_ratio = old_w / new_w

    for i in range(new_h):
        for j in range(new_w):
            # Map back to original coordinates
            src_i = i * row_ratio
            src_j = j * col_ratio
            
            if method == 'nearest':
                # (a) Nearest-neighbor: Round to closest integer [cite: 46]
                orig_i = min(int(round(src_i)), old_h - 1)
                orig_j = min(int(round(src_j)), old_w - 1)
                zoomed[i, j] = image[orig_i, orig_j]
                
            elif method == 'bilinear':
                # (b) Bilinear: Linear interpolation of 4 neighbors [cite: 47]
                i1, j1 = int(src_i), int(src_j)
                i2, j2 = min(i1 + 1, old_h - 1), min(j1 + 1, old_w - 1)
                
                # Fractional distances
                di, dj = src_i - i1, src_j - j1
                
                # Interpolation formula
                val = (1-di)*(1-dj)*image[i1,j1] + di*(1-dj)*image[i2,j1] + \
                      (1-di)*dj*image[i1,j2] + di*dj*image[i2,j2]
                zoomed[i, j] = int(val)
                
    return zoomed

def calculate_ssd(img1, img2):
    # Normalized SSD 
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    return np.sum(diff**2) / (img1.shape[0] * img1.shape[1])

# Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
large_img = cv2.imread(os.path.join(script_dir, '..', 'Images', 'im01.png'), 0)
small_img = cv2.imread(os.path.join(script_dir, '..', 'Images', 'im01small.png'), 0)

if large_img is not None and small_img is not None:
    # Zoom factor to match original size
    factor = large_img.shape[0] / small_img.shape[0]
    
    nearest_res = zoom_image(small_img, factor, 'nearest')
    bilinear_res = zoom_image(small_img, factor, 'bilinear')
    
    ssd_n = calculate_ssd(large_img, nearest_res)
    ssd_b = calculate_ssd(large_img, bilinear_res)
    
    print(f"SSD Nearest: {ssd_n:.4f}")
    print(f"SSD Bilinear: {ssd_b:.4f}")
    
    # Save results
    plt.figure(figsize=(12, 4))
    plt.subplot(131); plt.imshow(large_img, cmap='gray'); plt.title('Original Large')
    plt.subplot(132); plt.imshow(nearest_res, cmap='gray'); plt.title(f'Nearest (SSD: {ssd_n:.2f})')
    plt.subplot(133); plt.imshow(bilinear_res, cmap='gray'); plt.title(f'Bilinear (SSD: {ssd_b:.2f})')
    plt.savefig(os.path.join(script_dir, '..', 'Images', 'Results', 'q7_results.png'))
    plt.show()