# ---------------------------------------------
# Student Name: Your Name
# Roll No: Your Roll No
# Course: Image Processing & Computer Vision
# Assignment: Intelligent Image Enhancement System
# ---------------------------------------------

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim

print("Welcome to Intelligent Image Processing System")

# Create Output Folder
output_dir = "outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Task 2: Image Acquisition
img = cv2.imread("image.jpg")

if img is None:
    print("Error: image.jpg not found!")
    exit()

img = cv2.resize(img, (512, 512))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Save original & gray
cv2.imwrite(os.path.join(output_dir, "original.jpg"), img)
cv2.imwrite(os.path.join(output_dir, "gray.jpg"), gray)

# Task 3: Add Noise
# Gaussian Noise
gaussian_noise = np.random.normal(0, 25, gray.shape)
noisy_gaussian = gray + gaussian_noise
noisy_gaussian = np.clip(noisy_gaussian, 0, 255).astype(np.uint8)

# Salt & Pepper Noise
noisy_sp = gray.copy()
prob = 0.05
rand = np.random.rand(*gray.shape)
noisy_sp[rand < prob] = 0
noisy_sp[rand > 1 - prob] = 255

cv2.imwrite(os.path.join(output_dir, "noisy_gaussian.jpg"), noisy_gaussian)
cv2.imwrite(os.path.join(output_dir, "noisy_sp.jpg"), noisy_sp)

# Restoration Filters
mean_filter = cv2.blur(noisy_sp, (5, 5))
median_filter = cv2.medianBlur(noisy_sp, 5)
gaussian_filter = cv2.GaussianBlur(noisy_sp, (5, 5), 0)

cv2.imwrite(os.path.join(output_dir, "mean_filter.jpg"), mean_filter)
cv2.imwrite(os.path.join(output_dir, "median_filter.jpg"), median_filter)
cv2.imwrite(os.path.join(output_dir, "gaussian_filter.jpg"), gaussian_filter)

# Enhancement
equalized = cv2.equalizeHist(gray)
cv2.imwrite(os.path.join(output_dir, "enhanced.jpg"), equalized)

# Task 4: Segmentation
_, thresh_global = cv2.threshold(equalized, 127, 255, cv2.THRESH_BINARY)
_, thresh_otsu = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
dilation = cv2.dilate(thresh_otsu, kernel, iterations=1)
erosion = cv2.erode(thresh_otsu, kernel, iterations=1)

cv2.imwrite(os.path.join(output_dir, "threshold_global.jpg"), thresh_global)
cv2.imwrite(os.path.join(output_dir, "threshold_otsu.jpg"), thresh_otsu)
cv2.imwrite(os.path.join(output_dir, "dilation.jpg"), dilation)
cv2.imwrite(os.path.join(output_dir, "erosion.jpg"), erosion)

# Task 5: Edge Detection
edges = cv2.Canny(gray, 100, 200)
cv2.imwrite(os.path.join(output_dir, "edges.jpg"), edges)

# Contours
contours, _ = cv2.findContours(thresh_otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour_img = img.copy()

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(contour_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imwrite(os.path.join(output_dir, "contours.jpg"), contour_img)

# Feature Extraction (ORB)
orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(gray, None)
orb_img = cv2.drawKeypoints(img, keypoints, None)

cv2.imwrite(os.path.join(output_dir, "features_orb.jpg"), orb_img)

# Task 6: Performance Metrics
def mse(original, processed):
    return np.mean((original - processed) ** 2)

def psnr(original, processed):
    m = mse(original, processed)
    if m == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(m))

mse_val = mse(gray, equalized)
psnr_val = psnr(gray, equalized)
ssim_val = ssim(gray, equalized)

print("\n--- Performance Metrics ---")
print("MSE:", mse_val)
print("PSNR:", psnr_val)
print("SSIM:", ssim_val)

# Task 7: Visualization
titles = ['Original', 'Gray', 'Noisy', 'Restored', 'Enhanced', 'Segmented', 'Features']
images = [
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
    gray,
    noisy_sp,
    median_filter,
    equalized,
    thresh_otsu,
    orb_img
]

plt.figure(figsize=(15, 10))
for i in range(len(images)):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.savefig(os.path.join(output_dir, "final_output.png"))
plt.show()

print("\n✅ All output images saved in 'outputs/' folder successfully!")
print("✅ System executed successfully!")