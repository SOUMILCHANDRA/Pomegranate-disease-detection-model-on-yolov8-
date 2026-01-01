import cv2
import numpy as np
from ultralytics import YOLO

# Import functions from inference_jetson (assuming it's in the same dir)
try:
    from inference_jetson import get_infection_level
except ImportError:
    # Copy paste logic if import fails
    def get_infection_level(leaf_img):
        hsv = cv2.cvtColor(leaf_img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Yellow
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([30, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Brown/Black
        lower_brown = np.array([0, 20, 20])
        upper_brown = np.array([20, 255, 200])
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
        
        disease_mask = cv2.bitwise_or(mask_yellow, mask_brown)
        
        disease_pixels = cv2.countNonZero(disease_mask)
        green_pixels = cv2.countNonZero(mask_green)
        total_leaf_pixels = disease_pixels + green_pixels
        
        if total_leaf_pixels == 0:
            return 0.0
            
        return round((disease_pixels / total_leaf_pixels) * 100, 2)

from leaf_filter import is_pomegranate_leaf_candidate, is_pomegranate_fruit_candidate
from inference_jetson import get_fruit_infection_level, is_flower_candidate

def verify():
    # Testing User Image 25 (New Sample)
    img_path = r'e:/SIH2/test_image_25.jpg'
    frame = cv2.imread(img_path)
    if frame is None:
        print("Image not found")
        return

    # Load Model
    print("Loading Model...")
    try:
        model = YOLO(r"e:\SIH2\pomegranate_disease_model\yolov8n_cls_run_v2\weights\best.pt")
    except Exception as e:
        print(f"ERROR LOAING MODEL: {e}")
        return

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # ---------------------------------------------------------
    # 1. GREEN PIPELINE
    # ---------------------------------------------------------
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    kernel = np.ones((5,5), np.uint8)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=3) 
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours_green)} potential green contours")
    
    found_leaves = 0
    found_green_fruits = 0
    
    for cnt in contours_green:
        area = cv2.contourArea(cnt)
        if area < 1000: continue
        
        if is_pomegranate_leaf_candidate(cnt):
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
            found_leaves += 1
            
        elif is_pomegranate_fruit_candidate(cnt, strict=True):
             x, y, w, h = cv2.boundingRect(cnt)
             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
             cv2.putText(frame, "Green Fruit", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
             found_green_fruits += 1

    # ---------------------------------------------------------
    # 2. RED PIPELINE (Fruits/Flowers)
    # ---------------------------------------------------------
    lower_red1 = np.array([0, 20, 20])
    upper_red1 = np.array([35, 255, 255])
    lower_red2 = np.array([135, 20, 20])
    upper_red2 = np.array([180, 255, 255])
    lower_orange = np.array([10, 20, 20])
    upper_orange = np.array([35, 255, 255])
    lower_pink = np.array([145, 20, 20])
    upper_pink = np.array([160, 255, 255])
    
    mask_fruit = cv2.inRange(hsv, lower_red1, upper_red1) | \
                 cv2.inRange(hsv, lower_red2, upper_red2) | \
                 cv2.inRange(hsv, lower_orange, upper_orange) | \
                 cv2.inRange(hsv, lower_pink, upper_pink)
                 
    mask_fruit = cv2.morphologyEx(mask_fruit, cv2.MORPH_OPEN, kernel, iterations=1) 
    mask_fruit = cv2.morphologyEx(mask_fruit, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours_fruit, _ = cv2.findContours(mask_fruit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours_fruit)} potential red/orange fruit/flower contours")
    
    # Dynamic Area Threshold
    max_fruit_area = 0
    for cnt in contours_fruit:
         a = cv2.contourArea(cnt)
         if a > max_fruit_area: max_fruit_area = a
    dynamic_min_area = max(500, max_fruit_area * 0.05)
    print(f"DEBUG: Max Area={max_fruit_area}, Dynamic Min={dynamic_min_area}")
    
    found_red_fruits = 0
    found_flowers = 0
    
    for cnt in contours_fruit:
        area = cv2.contourArea(cnt)
        if area < dynamic_min_area: continue
        
        # Check Flower (Always)
        if is_flower_candidate(cnt):
             x, y, w, h = cv2.boundingRect(cnt)
             class_name = "Flower"
             if cv2.contourArea(cnt) > 3000: class_name = "Flower Cluster"
             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
             cv2.putText(frame, class_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
             found_flowers += 1
             continue

        # Check Red Fruit
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = min(w, h) / max(w, h) if w > 0 and h > 0 else 0
        hull = cv2.convexHull(cnt)
        solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
        
        # Potted Plant Fix: Reject Perfect Circles
        if ratio > 0.95: 
             print(f"DEBUG RED: Area={area}, Ratio={ratio:.3f} > 0.95 (REJECTED CIRCLE)")
             continue
             
        # Shape Check (Relaxed)
        can = is_pomegranate_fruit_candidate(cnt, strict=False)
        print(f"DEBUG RED: Area={area}, Ratio={ratio:.3f}, Solidity={solidity:.3f}, Accepted={can}")
        
        if can:
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0: continue
            
            try:
                results = model(roi, verbose=False)
                probs = results[0].probs
                top1_index = probs.top1
                class_name = model.names[top1_index]
                confidence = probs.top1conf.item()
                print(f"DEBUG: Class={class_name}, Confidence={confidence:.3f}")
                
                # Filter Low Confidence
                # Dynamic Threshold: Large Fruits (Clear View) might have lower model confidence due to zoom/texture
                # but are geometrically likely to be fruits. Small blobs need high confidence.
                min_conf = 0.25 if area > 50000 else 0.45
                
                if confidence < min_conf: continue
                
                # Infection
                roi_hsv = hsv[y:y+h, x:x+w]
                infection_level, severity = get_fruit_infection_level(roi_hsv, None)
                
                label = f"{class_name} {confidence*100:.0f}%"
                if infection_level > 0: 
                    label += f" | Inf:{infection_level}%" 
                
                print(f"  -> LABEL: {label}")
                
                # Draw
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                found_red_fruits += 1
            except Exception as e:
                print(f"  -> Model Error: {e}")

    print(f"Kept {found_leaves} leaves, {found_green_fruits} green fruits, {found_red_fruits} red fruits, {found_flowers} flowers.")
    cv2.imwrite(r'e:/SIH2/verification_result.jpg', frame)
    print(r"Saved result to e:/SIH2/verification_result.jpg")

if __name__ == "__main__":
    verify()
