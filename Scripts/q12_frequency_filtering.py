# 258799R - Manuel I.T
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def homomorphic_filter(image, d0=30, rl=0.5, rh=2.0, c=1):
    # 1. Log Transformation to separate illumination and reflectance
    # We use log1p to avoid log(0) issues
    img_log = np.log1p(np.array(image, dtype="float"))
    
    # 2. Compute 2D FFT and Shift
    dft = np.fft.fft2(img_log)
    dft_shift = np.fft.fftshift(dft)
    
    # 3. Create High-Pass Filter Mask (Gaussian-based Homomorphic)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    x = np.linspace(-ccol, cols - ccol - 1, cols)
    y = np.linspace(-crow, rows - crow - 1, rows)
    X, Y = np.meshgrid(x, y)
    
    # Distance squared from center
    D_sq = X**2 + Y**2
    
    # Homomorphic transfer function: H(u,v) = (rh - rl) * (1 - exp(-c*(D^2/D0^2))) + rl
    H = (rh - rl) * (1 - np.exp(-c * (D_sq / (d0**2)))) + rl

    # 4. Apply Mask 
    fshift = dft_shift * H
    
    # 5. Inverse FFT 
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    
    # 6. Exponential transformation (reverse of log)
    img_exp = np.expm1(np.real(img_back))
    
    # Normalize to 0-255 range
    img_res = np.clip(img_exp, 0, 255).astype(np.uint8)
    return img_res

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, '..', 'Images', 'runway.png')
    results_dir = os.path.join(script_dir, '..', 'Images', 'Results')
    os.makedirs(results_dir, exist_ok=True)
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return

    # Apply homomorphic filter
    filtered_img = homomorphic_filter(img, d0=30, rl=0.5, rh=1.5)

    # Visualization
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title('Original (Uneven Lighting)')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(filtered_img, cmap='gray')
    plt.title('Homomorphic Filtered')
    plt.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'q12_homomorphic_results.png')
    plt.savefig(save_path)
    print(f"Results saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()