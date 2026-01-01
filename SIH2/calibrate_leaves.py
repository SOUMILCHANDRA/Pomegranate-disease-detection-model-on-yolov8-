import cv2
import numpy as np
import glob
import os

def analyze_leaves(img_path, label):
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Error reading {img_path}")
        return

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Green Mask
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ratios = []
    solidities = []
    
    print(f"\n--- Analyzing {label} ({os.path.basename(img_path)}) ---")
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500: continue # Ignore noise
        
        # Geometry
        x, y, w, h = cv2.boundingRect(cnt)
        rect = cv2.minAreaRect(cnt)
        width = rect[1][0]
        height = rect[1][1]
        
        if width == 0 or height == 0: continue
        ratio = min(width, height) / max(width, height)
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        solidity = area / hull_area
        
        ratios.append(ratio)
        solidities.append(solidity)
        print(f"Leaf: Area={area:.0f}, Ratio={ratio:.3f}, Solidity={solidity:.3f}")

    if ratios:
        print(f"STATS -> Avg Ratio: {np.mean(ratios):.3f}, Avg Solidity: {np.mean(solidities):.3f}")
    else:
        print("No valid green contours found.")

# 1. Analyze User Uploaded "Real" Leaves
uploaded_images = glob.glob(r"C:\Users\Admin\.gemini\antigravity\brain\1c84d50b-0878-4f9f-a477-c280fb5ea98b\uploaded_image_*.jpg")
for img in uploaded_images:
    analyze_leaves(img, "REAL POMEGRANATE")

# 2. Analyze "Imposter" Image 24
analyze_leaves(r"e:\SIH2\test_image_24.jpg", "IMPOSTER (Image 24)")
