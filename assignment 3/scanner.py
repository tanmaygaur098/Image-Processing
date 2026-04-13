"""
Name: Tanmay Gaur
Roll No: 2301010419
Course: Image Processing & Computer Vision
Assignment: Medical Image Compression & Segmentation
"""

import cv2
import numpy as np
import os

# Utility Functions

def create_output_folder():
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
        print("✔ 'outputs/' folder created")


def load_image(path):
    if not os.path.exists(path):
        raise Exception(f"❌ Image not found at: {path}")

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise Exception("❌ Error loading image!")

    return img

# Task 1: RLE Compression

def rle_encode(image):
    pixels = image.flatten()
    encoding = []

    prev = pixels[0]
    count = 1

    for pixel in pixels[1:]:
        if pixel == prev:
            count += 1
        else:
            encoding.append((prev, count))
            prev = pixel
            count = 1

    encoding.append((prev, count))
    return encoding


def compression_stats(original, encoded):
    original_size = original.size
    encoded_size = len(encoded) * 2  # (value, count)

    ratio = original_size / encoded_size
    savings = (1 - encoded_size / original_size) * 100

    return ratio, savings

# Task 2: Segmentation

def global_threshold(image):
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return thresh


def otsu_threshold(image):
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# Task 3: Morphology

def morphology(image):
    kernel = np.ones((3, 3), np.uint8)

    dilation = cv2.dilate(image, kernel, iterations=1)
    erosion = cv2.erode(image, kernel, iterations=1)

    return dilation, erosion

# Main Pipeline

def process_image(image_path):
    print("\n--- Processing Image ---")

    img = load_image(image_path)
    cv2.imwrite("outputs/original.png", img)
    print("✔ Image loaded")

    # ---------- Compression ----------
    print("\n--- Compression (RLE) ---")

    encoded = rle_encode(img)
    ratio, savings = compression_stats(img, encoded)

    print(f"Compression Ratio: {ratio:.2f}")
    print(f"Storage Savings: {savings:.2f}%")

    # Save RLE output (partial for readability)
    with open("outputs/rle.txt", "w") as f:
        for val, count in encoded[:1000]:
            f.write(f"{val}:{count} ")

    print("✔ RLE saved")

    # ---------- Segmentation ----------
    print("\n--- Segmentation ---")

    global_seg = global_threshold(img)
    otsu_seg = otsu_threshold(img)

    cv2.imwrite("outputs/global_threshold.png", global_seg)
    cv2.imwrite("outputs/otsu_threshold.png", otsu_seg)

    print("✔ Segmentation done")

    # ---------- Morphology ----------
    print("\n--- Morphological Processing ---")

    g_dil, g_ero = morphology(global_seg)
    o_dil, o_ero = morphology(otsu_seg)

    cv2.imwrite("outputs/global_dilation.png", g_dil)
    cv2.imwrite("outputs/global_erosion.png", g_ero)
    cv2.imwrite("outputs/otsu_dilation.png", o_dil)
    cv2.imwrite("outputs/otsu_erosion.png", o_ero)

    print("✔ Morphology done")

    return ratio, savings

# MAIN

if __name__ == "__main__":
    print("🚀 Script started")

    create_output_folder()

    # ✅ IMPORTANT: your image name
    image_path = "image.jpg"

    try:
        ratio, savings = process_image(image_path)

        print("\n FINAL RESULTS ")
        print(f"Compression Ratio: {ratio:.2f}")
        print(f"Storage Savings: {savings:.2f}%")

        print("\n✔ All outputs saved in 'outputs/' folder")

    except Exception as e:
        print(e)
