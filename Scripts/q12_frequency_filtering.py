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
    os.makedirs(results_dir, exist_ok=True)
    
    # Load as grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not load image.")
        return

    # 2. Compute 2D FFT and Shift
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2 # Center coordinates

    # 3. Create Gaussian Low Pass Filter (GLPF) Mask
    # D0 is the cutoff frequency (adjust this to change blur level)
    D0 = 30 
    x = np.linspace(-ccol, cols - ccol - 1, cols)
    y = np.linspace(-crow, rows - crow - 1, rows)
    X, Y = np.meshgrid(x, y)
    
    # Gaussian Mask Formula: H(u,v) = exp(-D^2 / (2 * D0^2))
    mask = np.exp(-(X**2 + Y**2) / (2 * D0**2))

    # 4. Apply Mask (Element-wise Multiplication)
    fshift = dft_shift * mask
    
    # 5. Inverse FFT to return to Spatial Domain
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # 6. Visualization for Report
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    # Use log transform to visualize the masked spectrum
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(f'Filtered Spectrum (D0={D0})')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(img_back, cmap='gray')
    plt.title('Result (Spatial Domain)')
    plt.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'q12_frequency_filter_results.png')
    plt.savefig(save_path)
    print(f"Frequency filtering results saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()