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
    
    # Load as Grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load {img_path}")
        return

    # --- (b) Constant Addition ---
    # Increases overall brightness
    added = cv2.add(img, 60)

    # --- (c) Constant Multiplication ---
    # Increases contrast by scaling intensities
    multiplied = cv2.multiply(img, 1.4)

    # --- (d) Image Blending ---
    # Blend the rice image with a horizontally flipped version
    img_flipped = cv2.flip(img, 1)
    blended = cv2.addWeighted(img, 0.5, img_flipped, 0.5, 0)

    # 4. Visualization for Report
    plt.figure(figsize=(16, 8))
    
    plt.subplot(221); plt.imshow(img, cmap='gray'); plt.title('Original (Grayscale)'); plt.axis('off')
    plt.subplot(222); plt.imshow(added, cmap='gray'); plt.title('(b) Addition (+60)'); plt.axis('off')
    plt.subplot(223); plt.imshow(multiplied, cmap='gray'); plt.title('(c) Multiplication (1.4x)'); plt.axis('off')
    plt.subplot(224); plt.imshow(blended, cmap='gray'); plt.title('(d) 50/50 Blending'); plt.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'q10_grayscale_arithmetic.png')
    plt.savefig(save_path)
    print(f"Grayscale results saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()