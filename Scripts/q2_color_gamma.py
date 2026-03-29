#258799R - Manuel I.T
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Load Image
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, '..', 'Images', 'highlights_and_shadows.jpg') # Fig 2
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. Convert to Lab color space
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
l, a, b = cv2.split(img_lab)

# 3. Apply Gamma Correction to L channel
gamma = 0.6  # State this value in your report
l_norm = l / 255.0
l_gamma = np.power(l_norm, gamma)
l_corrected = (l_gamma * 255).astype(np.uint8)

# 4. Merge back and convert to RGB
img_lab_corrected = cv2.merge([l_corrected, a, b])
img_corrected_rgb = cv2.cvtColor(img_lab_corrected, cv2.COLOR_Lab2RGB)

# 5. Visualization for Report
plt.figure(figsize=(12, 8))

# Original and Corrected Images
plt.subplot(2, 2, 1); plt.imshow(img_rgb); plt.title('Original'); plt.axis('off')
plt.subplot(2, 2, 2); plt.imshow(img_corrected_rgb); plt.title(f'Gamma Corrected (γ={gamma})'); plt.axis('off')

# Histograms
plt.subplot(2, 2, 3); plt.hist(l.flatten(), bins=256, color='gray'); plt.title('Original L Histogram')
plt.subplot(2, 2, 4); plt.hist(l_corrected.flatten(), bins=256, color='blue'); plt.title('Corrected L Histogram')

plt.tight_layout()
plt.savefig(os.path.join(script_dir, '..', 'Images', 'Results', 'q2_results.png'))
plt.show()