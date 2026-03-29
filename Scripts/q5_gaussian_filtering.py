#258799R - Manuel I.T
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def get_gaussian_kernel(size, sigma):
    # (a) Compute normalized 5x5 Gaussian kernel using NumPy [cite: 26]
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def main():
    # Setup absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, '..', 'Images', 'runway.png')
    results_dir = os.path.join(script_dir, '..', 'Images', 'Results')
    os.makedirs(results_dir, exist_ok=True)

    # (a) 5x5 Kernel
    kernel_5x5 = get_gaussian_kernel(5, 2)
    
    # (b) 51x51 Kernel 3D Plot [cite: 27]
    kernel_51x51 = get_gaussian_kernel(51, 2)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(np.arange(51), np.arange(51))
    ax.plot_surface(x, y, kernel_51x51, cmap='viridis')
    plt.title('51x51 Gaussian Kernel ($\sigma=2$)')
    plt.savefig(os.path.join(results_dir, 'q5_3d_plot.png'))

    # Load and check image [cite: 28]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not find {img_path}")
        return

    # (c) Manual Smoothing [cite: 28]
    manual_blur = cv2.filter2D(img, -1, kernel_5x5)

    # (d) OpenCV Smoothing [cite: 29]
    opencv_blur = cv2.GaussianBlur(img, (5, 5), 2)

    # Final Comparison Plot 
    plt.figure(figsize=(15, 5))
    plt.subplot(131); plt.imshow(img, cmap='gray'); plt.title('Original')
    plt.subplot(132); plt.imshow(manual_blur, cmap='gray'); plt.title('Manual Blur')
    plt.subplot(133); plt.imshow(opencv_blur, cmap='gray'); plt.title('OpenCV Blur')
    
    # Save before showing
    plt.savefig(os.path.join(results_dir, 'q5_comparison.png'))
    plt.show()

if __name__ == "__main__":
    main()