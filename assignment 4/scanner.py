"""
Name: Tanmay Gaur
Roll No: 2301010419
Course: Image Processing & Computer Vision
Assignment: Feature-Based Traffic Monitoring System
"""

import cv2
import numpy as np
import os

# Utility

def create_output_folder():
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
        print("✔ outputs/ folder created")


def load_image(path):
    if not os.path.exists(path):
        raise Exception(" Image not found!")

    img = cv2.imread(path)
    if img is None:
        raise Exception(" Error loading image!")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

# Task 1: Edge Detection

def sobel_edges(gray):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.uint8(np.clip(sobel, 0, 255))
    return sobel


def canny_edges(gray):
    return cv2.Canny(gray, 100, 200)

# Task 2: Contours & Objects

def detect_contours(image, edge_img):
    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = image.copy()

    print("\n--- Object Measurements ---")

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if area > 100:  # filter noise
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)

            print(f"Object {i+1}: Area = {area:.2f}, Perimeter = {perimeter:.2f}")

    return output

# Task 3: Feature Extraction

def orb_features(gray, image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    output = cv2.drawKeypoints(image, keypoints, None, color=(0,255,0))
    print(f"\n✔ ORB Keypoints detected: {len(keypoints)}")

    return output

# Pipeline

def process_image(path):
    print("\n--- Processing Traffic Image ---")

    img, gray = load_image(path)
    cv2.imwrite("outputs/original.png", img)

    print("✔ Image loaded")

    # ---------- Edge Detection ----------
    print("\n--- Edge Detection ---")

    sobel = sobel_edges(gray)
    canny = canny_edges(gray)

    cv2.imwrite("outputs/sobel.png", sobel)
    cv2.imwrite("outputs/canny.png", canny)

    print("✔ Sobel & Canny done")

    # ---------- Contours ----------
    print("\n--- Contour Detection ---")

    contour_img = detect_contours(img, canny)
    cv2.imwrite("outputs/contours.png", contour_img)

    print("✔ Contours & bounding boxes done")

    # ---------- Feature Extraction ----------
    print("\n--- Feature Extraction (ORB) ---")

    orb_img = orb_features(gray, img)
    cv2.imwrite("outputs/orb_features.png", orb_img)

    print("✔ Feature extraction done")

# MAIN

if __name__ == "__main__":
    print("🚀 Script started")

    create_output_folder()

    # 👉 your image name
    image_path = "image.jpg"

    try:
        process_image(image_path)

        print("\n DONE ")
        print("✔ Outputs saved in 'outputs/' folder")

    except Exception as e:
        print(e)
