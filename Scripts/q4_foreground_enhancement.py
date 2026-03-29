#258799R - Manuel I.T
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Load Image
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, '..', 'Images', 'highlights_and_shadows.jpg') # Fig 3
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Otsu's Thresholding 
# We invert the threshold (THRESH_BINARY_INV) so foreground (dark) is 255 (white)
val, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
print(f"Otsu's Threshold Value: {val}") # Report this value 

# 3. Histogram Equalization on Foreground only 
# Apply global equalization to the whole image first
equalized_full = cv2.equalizeHist(gray)

# Use the mask to combine: Equalized for foreground, Original for background
# mask > 0 selects the woman/room
result = np.where(mask > 0, equalized_full, gray)

# 4. Visualization for Report 
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(gray, cmap='gray'); plt.title('Original Grayscale')
plt.subplot(1, 3, 2); plt.imshow(mask, cmap='gray'); plt.title('Otsu Mask')
plt.subplot(1, 3, 3); plt.imshow(result, cmap='gray'); plt.title('Foreground Enhanced')

# Save result
save_path = os.path.join(script_dir, '..', 'Images', 'Results', 'q4_results.png')
plt.savefig(save_path)
plt.show()