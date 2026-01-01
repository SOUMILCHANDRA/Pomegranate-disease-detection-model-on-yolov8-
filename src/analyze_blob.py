import cv2
import numpy as np

def inspect_blobs(img_path):
    frame = cv2.imread(img_path)
    if frame is None: return

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Fruit Mask (Same as main_analyzer)
    lower_red1 = np.array([0, 20, 20])
    upper_red1 = np.array([35, 255, 255])
    lower_red2 = np.array([135, 20, 20])
    upper_red2 = np.array([180, 255, 255])
    lower_orange = np.array([10, 20, 20])
    upper_orange = np.array([35, 255, 255])
    
    lower_pink = np.array([145, 20, 20])
    upper_pink = np.array([160, 255, 255])
    
    mask = cv2.inRange(hsv, lower_red1, upper_red1) | \
           cv2.inRange(hsv, lower_red2, upper_red2) | \
           cv2.inRange(hsv, lower_orange, upper_orange) | \
           cv2.inRange(hsv, lower_pink, upper_pink)
           
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    with open("e:/SIH2/blob_metrics.txt", "w") as f:
        print(f"--- Inspecting {img_path} ---")
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100: continue
            
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = min(w, h) / max(w, h)
            
            log_line = f"Blob: Area={area:.0f}, Solidity={solidity:.3f}, Ratio={ratio:.3f}, BBox=({x},{y},{w},{h})"
            print(log_line)
            f.write(log_line + "\n")
            
            # Draw it for visual check (saved to debug_blob.jpg)
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imwrite("debug_blob.jpg", frame)

inspect_blobs(r"e:/SIH2/test_image_34.png")
