import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # 1. Setup paths relative to this script's location (Scripts folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, '..', 'Images', 'runway.png')
    results_dir = os.path.join(script_dir, '..', 'Images', 'Results')
    
    # Ensure the Results directory exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 2. Load the runway image in grayscale [cite: 4, 5]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error: Could not load image at {img_path}")
        return

    # 3. Normalize input pixel intensity 'r' to range [0, 1] 
    r = img.astype(np.float32) / 255.0

    # --- Transformation (a): Gamma correction with γ = 0.5 --- [cite: 6]
    s_a = np.power(r, 0.5)

    # --- Transformation (b): Gamma correction with γ = 2 --- [cite: 7]
    s_b = np.power(r, 2.0)

    # --- Transformation (c): Contrast Stretching (Piecewise Linear) --- [cite: 8]
    # r1 = 0.2, r2 = 0.8 [cite: 11, 12]
    r1, r2 = 0.2, 0.8
    s_c = np.where(r < r1, 0, 
                  np.where(r > r2, 1, 
                  (r - r1) / (r2 - r1)))

    # 4. Convert results back to 8-bit (0-255) for visualization
    res_a = (s_a * 255).astype(np.uint8)
    res_b = (s_b * 255).astype(np.uint8)
    res_c = (s_c * 255).astype(np.uint8)

    # 5. Visualization and Saving for the Report [cite: 94, 96]
    titles = ['Original Runway', 'Gamma=0.5', 'Gamma=2.0', 'Contrast Stretching']
    images = [img, res_a, res_b, res_c]

    plt.figure(figsize=(16, 10))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save results to the folder shown in your screenshot
    save_path = os.path.join(results_dir, 'q1_runway_transformations.png')
    plt.savefig(save_path)
    print(f"Successfully saved results to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()