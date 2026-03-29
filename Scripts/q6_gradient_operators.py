# 258799R - Manuel I.T 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def get_dog_kernels(size, sigma):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    
    # Standard Gaussian 
    g = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    g /= np.sum(g) # Normalize base Gaussian
    
    # (b) Derivatives of Gaussian [cite: 38, 39, 40]
    kernel_x = -(xx / (sigma**2)) * g
    kernel_y = -(yy / (sigma**2)) * g
    
    return kernel_x, kernel_y

# (b) Compute 5x5 kernels for sigma = 2 [cite: 40]
kx, ky = get_dog_kernels(5, 2)

# (c) Visualize 51x51 kernel (x-direction) [cite: 41]
kx_large, _ = get_dog_kernels(51, 2)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.arange(51), np.arange(51))
ax.plot_surface(X, Y, kx_large, cmap='RdBu')
ax.set_title('51x51 Derivative-of-Gaussian (x) Surface Plot')
plt.show()

# (d) & (e) Apply Gradients and Compare 
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, '..', 'Images', 'runway.png')
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is not None:
    # Manual DoG Gradients [cite: 42]
    grad_x = cv2.filter2D(img, cv2.CV_64F, kx)
    grad_y = cv2.filter2D(img, cv2.CV_64F, ky)

    # OpenCV Sobel Gradients (ksize=5 to match our 5x5 DoG) [cite: 43]
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    # Plotting for Comparison [cite: 96]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: X-Gradients
    axes[0,0].imshow(img, cmap='gray'); axes[0,0].set_title('Original')
    axes[0,1].imshow(np.abs(grad_x), cmap='gray'); axes[0,1].set_title('Manual DoG X')
    axes[0,2].imshow(np.abs(sobel_x), cmap='gray'); axes[0,2].set_title('OpenCV Sobel X')
    
    # Row 2: Y-Gradients
    axes[1,0].axis('off') # Spacer
    axes[1,1].imshow(np.abs(grad_y), cmap='gray'); axes[1,1].set_title('Manual DoG Y')
    axes[1,2].imshow(np.abs(sobel_y), cmap='gray'); axes[1,2].set_title('OpenCV Sobel Y')
    
    plt.tight_layout()
    # Save to the Results folder in your structure
    save_path = os.path.join(script_dir, '..', 'Images', 'Results', 'q6_comparison.png')
    plt.savefig(save_path)
    print(f"Results saved to {save_path}")
    plt.show()