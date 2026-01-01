import cv2
import numpy as np

def debug_masks(img_path):
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Could not read {img_path}")
        return

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Current Masks (from main_analyzer.py)
    lower_red1 = np.array([0, 20, 20])
    upper_red1 = np.array([35, 255, 255])
    
    lower_red2 = np.array([135, 20, 20])
    upper_red2 = np.array([180, 255, 255])
    
    lower_orange = np.array([10, 20, 20])
    upper_orange = np.array([35, 255, 255])
    
    lower_pink = np.array([145, 20, 20])
    upper_pink = np.array([160, 255, 255])
    
    # 1. Generate Individual Masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask3 = cv2.inRange(hsv, lower_orange, upper_orange)
    mask4 = cv2.inRange(hsv, lower_pink, upper_pink)
    
    combined_mask = mask1 | mask2 | mask3 | mask4
    
    # Clean up
    kernel = np.ones((5,5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Save for visual inspection
    cv2.imwrite("debug_mask_combined.jpg", combined_mask)
    
    # Overlay on original
    result = cv2.bitwise_and(frame, frame, mask=combined_mask)
    cv2.imwrite("debug_overlay.jpg", result)
    print(f"Saved debug_mask_combined.jpg and debug_overlay.jpg for {img_path}")

# Run on the problematic image (Image 33) and the first one (Image 32)
debug_masks(r"e:/SIH2/test_image_33.png")
