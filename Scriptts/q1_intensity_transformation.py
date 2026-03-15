#258799R Q1: Intensity Transformation

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Coordinate points from the graph in Fig 1a
input_values =  [0, 100, 150, 255]
output_values = [0,  50, 200, 255]

def solve_q1():
    # --- PATH ADJUSTMENTS ---
    input_path = 'Images/emma.jpg'
    output_dir = 'Images/Results'  # Subfolder inside Images
    output_filename = 'q1_transformed_emma.jpg'
    output_path = os.path.join(output_dir, output_filename)
    
    # Check if the Results folder exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # -------------------------

    # 2. Load the image in grayscale
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error: Could not find {input_path}. Please check the folder name and filename.")
        return

    # 3. Create the Lookup Table (LUT)
    all_inputs = np.arange(256)
    lut_values = np.interp(all_inputs, input_values, output_values).astype(np.uint8)

    # 4. Apply the transformation
    transformed_img = cv2.LUT(img, lut_values)

    # 5. Save the result to the subfolder
    cv2.imwrite(output_path, transformed_img)
    print(f"Success! Transformed image saved as: {output_path}")

    # 6. Visualization
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original (Emma)')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.plot(input_values, output_values, 'r-o', linewidth=2)
    plt.title('Transformation Graph (Fig 1a)')
    plt.xlabel('Input Intensity')
    plt.ylabel('Output Intensity')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.imshow(transformed_img, cmap='gray')
    plt.title('Output Result')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    solve_q1()