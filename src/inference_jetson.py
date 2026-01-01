import cv2
import numpy as np
from ultralytics import YOLO
import time

def get_infection_level(leaf_img):
    """
    Calculate infection level based on color segmentation within the leaf.
    Assumes leaf_img is a cropped image of the leaf.
    """
    hsv = cv2.cvtColor(leaf_img, cv2.COLOR_BGR2HSV)
    
    # Define range for "healthy" green part
    # Adjust these values based on actual lighting/leaf color
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    
    # Mask for green area (healthy)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Total leaf area (approximate by non-black pixels if background is black, 
    # or just use the whole contour area from the parent step)
    # Here we assume leaf_img might have some background. 
    # Better approach: Use the mask from detection step.
    
    # Let's count non-zero pixels in the leaf (assuming we passed a masked image or rectangular crop)
    # Simple approximation: Disease = Non-Green AND Non-Background
    # For a rectangular crop, this is hard without a mask.
    
    # alternative: Mask for "Disease Colors" (Brown, Yellow, Black spots)
    # Yellow
    lower_yellow = np.array([15, 50, 50])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Brown/Black (Low saturation/value or orange-ish hue)
    lower_brown = np.array([0, 20, 20])
    upper_brown = np.array([20, 255, 200])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    
    disease_mask = cv2.bitwise_or(mask_yellow, mask_brown)
    
    # Refine: exclude background (black) if present
    # If we assume the crop is just the bounding box, it has background. 
    # We really need the leaf mask to be accurate.
    
    disease_pixels = cv2.countNonZero(disease_mask)
    green_pixels = cv2.countNonZero(mask_green)
    
    total_leaf_pixels = disease_pixels + green_pixels
    
    if total_leaf_pixels == 0:
        return 0.0
        
    infection_level = (disease_pixels / total_leaf_pixels) * 100
    return round(infection_level, 2)

def detect_growth_phase(frame):
    """
    Analyzes frame to determine: 'FLOWERING', 'YOUNG', or 'MATURE'.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 1. Check for Flowers (Red/Orange/Yellow/Pink, Small/Irregular)
    lower_flower1 = np.array([0, 50, 50])
    upper_flower1 = np.array([35, 255, 255])
    lower_flower2 = np.array([145, 50, 50])
    upper_flower2 = np.array([180, 255, 255])
    
    mask_flower = cv2.inRange(hsv, lower_flower1, upper_flower1) | cv2.inRange(hsv, lower_flower2, upper_flower2)
    contours, _ = cv2.findContours(mask_flower, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    flower_candidates = 0
    fruit_candidates = 0
    
    # Measure Green for context
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_candidates = len(contours_green)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Determine if Flower or Fruit
        if area > 2500: 
            # Large object -> Likely Fruit or Cluster
            # If it's a Fruit (Round), count it
            if is_pomegranate_fruit_candidate(cnt, strict=False):
                fruit_candidates += 1
            # If it's irregular (Cluster), treat as Flower? 
            # Or just ignore for Phase?
            # User dislikes "Flowering" phase if fruits are present.
            # So let's bias towards Fruit logic if large.
            continue 
        
        if area > 1000:
            if is_flower_candidate(cnt):
                 flower_candidates += 1
            elif is_pomegranate_fruit_candidate(cnt, strict=False): 
                 fruit_candidates += 1
            else:
                 pass
            
    # DEBUG PRINT (Safe Version)
    # print(f"DEBUG PHASE: Green={green_candidates}, Flower={flower_candidates}, Fruit={fruit_candidates}")

    # HEURISTIC RULES
    if flower_candidates > 0:
        if flower_candidates > (fruit_candidates * 2): 
            return "FLOWERING"
        if fruit_candidates < 2 and flower_candidates >= 1:
            return "FLOWERING"
            
    # MATURE vs YOUNG (Color Based)
    # If we have Red Fruits, check their color intensity.
    # Count of 'leaf' candidates is unreliable (always high).
    
    if fruit_candidates > 0:
        # Sample the red mask to see if it's Deep Red (Mature) or Orange/Light (Young)
        # We can re-use the masks defined above or just call detect_fruit_phase on the whole frame (masked)
        # Faster approach: Check pixel ratio in the Fruit Mask.
        
        # Red1 (Deep Red + Red-Orange)
        lower_mat1 = np.array([0, 80, 50]) # Lowered Saturation slightly too
        upper_mat1 = np.array([20, 255, 255]) # Extended to 20
        lower_mat2 = np.array([160, 80, 50])
        upper_mat2 = np.array([180, 255, 255])
        
        # Orange/Yellow (Young/Turning)
        lower_young = np.array([21, 40, 50]) # Starts at 21
        upper_young = np.array([35, 255, 255])
        
        mask_mature = cv2.inRange(hsv, lower_mat1, upper_mat1) | cv2.inRange(hsv, lower_mat2, upper_mat2)
        mask_young = cv2.inRange(hsv, lower_young, upper_young)
        
        mat_pixels = cv2.countNonZero(mask_mature)
        young_pixels = cv2.countNonZero(mask_young)
        
        # print(f"DEBUG PHASE COLOR: Mature={mat_pixels}, Young={young_pixels}")
        
        # If we have significant Mature Red pixels
        # Bias towards MATURE if Mature > Young OR if Mature is significant (>30% of total fruit pixels)
        # Because trees often have mix, but if Red is present, it's considered Mature stage.
        if mat_pixels > young_pixels:
            return "MATURE"
        elif mat_pixels > (young_pixels * 0.5): # Even if Young is more, if Mature is substantial -> MATURE
             return "MATURE"
        else:
             return "YOUNG"

    # Fallback if no fruits found (e.g. very early stage)
    if green_candidates > fruit_candidates:
        return "YOUNG"
        
    return "MATURE"

def is_flower_candidate(contour):
    """
    Validates Pomegranate Flower (Vase/Urn shape, crinkled petals).
    """
    area = cv2.contourArea(contour)
    # Increased min area to 1500 to avoid noise/weed flowers
    if area < 1500: return False
    if area > 6000: return False # Reject large canopy/branch sections being called "Flowers"
    
    # Solidity: Low due to crinkled petals / stamens
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0: return False
    solidity = float(area) / hull_area
    
    # Flowers are less solid than fruits
    # Allow 0.5 to 0.75 (Tightened from 0.78 to reject leaves)
    if solidity < 0.50: return False
    
    # Reject very solid objects (likely fruits or trash)
    if solidity > 0.75: return False
    
    # Aspect Ratio: Vase shape might be elongated
    rect = cv2.minAreaRect(contour)
    (center), (width, height), angle = rect
    if width == 0 or height == 0: return False
    ratio = min(width, height) / max(width, height)
    
    # Leaves are narrow (Ratio 0.2 - 0.5). Flowers are broader/rounder.
    # Restored to 0.50 to reject Red Leaves (False Positives).
    if ratio < 0.50: return False
    
    # Roundness check: If it's round (Ratio > 0.6), it must be irregular.
    # Logic Update for Large Objects (Crude Fruits vs Clusters):
    # If roundness > 0.75, it's too round to be a flower, even if hollow.
    if ratio > 0.75: return False
    
    threshold = 0.60
    if area > 2000:
        threshold = 0.55 # Stricter for large objects to avoiding calling crude fruits "Flowers"
        
    if ratio > 0.60 and solidity > threshold:
        return False
    
    print(f"DEBUG FLOWER ACCEPT: Area={area}, Ratio={ratio:.3f}, Solidity={solidity:.3f}")
    return True

def detect_fruit_phase(fruit_img):
    """
    Analyzes the fruit ROI color profile to determine growth phase.
    Returns: 'YOUNG' or 'MATURE'.
    """
    hsv = cv2.cvtColor(fruit_img, cv2.COLOR_BGR2HSV)
    
    # Measure presence of "Mature Red" (Deep/Vibrant Red)
    # Hue 0-10, 160-180. Sat > 100. Val > 50.
    lower_mat1 = np.array([0, 100, 50])
    upper_mat1 = np.array([10, 255, 255])
    lower_mat2 = np.array([160, 100, 50])
    upper_mat2 = np.array([180, 255, 255])
    mask_mature = cv2.inRange(hsv, lower_mat1, upper_mat1) | cv2.inRange(hsv, lower_mat2, upper_mat2)
    
    # Measure presence of "Young Orange/Pink"
    # Hue 11-35 (Orange/Yellowish). Sat > 40.
    lower_young = np.array([11, 40, 50])
    upper_young = np.array([35, 255, 255])
    mask_young = cv2.inRange(hsv, lower_young, upper_young)
    
    mature_pixels = cv2.countNonZero(mask_mature)
    young_pixels = cv2.countNonZero(mask_young)
    
    # Heuristic: If we have significant orange/yellow compared to deep red -> Young
    total_color = mature_pixels + young_pixels
    if total_color == 0: return 'MATURE' # Default
    
    if young_pixels > mature_pixels * 0.5: # If Young is significant (>33% of color)
        return 'YOUNG'
    return 'MATURE'

def get_fruit_infection_level(fruit_img, phase=None):
    """
    Calculate infection level with Adaptive Auto-Phase Logic.
    """
    # 1. Smart Scan: Detect Phase if not provided
    if phase is None:
        phase = detect_fruit_phase(fruit_img)
        print(f"DEBUG: Internal Detected Phase: {phase}")
    
    hsv = cv2.cvtColor(fruit_img, cv2.COLOR_BGR2HSV)
    
    # 2. Adaptive Healthy Mask
    if phase == 'YOUNG':
        # Broad Range: Red + Orange + Pink + GREEN
        # Green Hue: 35-85
        lower_healthy1 = np.array([0, 40, 40])
        upper_healthy1 = np.array([25, 255, 255]) # Includes Orange (0-25)
        lower_healthy2 = np.array([155, 40, 40])
        upper_healthy2 = np.array([180, 255, 255]) # Includes Pink
        lower_healthy3 = np.array([30, 40, 40]) # Green
        upper_healthy3 = np.array([90, 255, 255]) 
        
        mask_healthy = cv2.inRange(hsv, lower_healthy1, upper_healthy1) | cv2.inRange(hsv, lower_healthy2, upper_healthy2) | cv2.inRange(hsv, lower_healthy3, upper_healthy3)
    else: # MATURE
        # Strict Range: Deep Red Only (Broader: 0-20)
        # Excludes Yellow (>20)
        lower_healthy1 = np.array([0, 60, 50])
        upper_healthy1 = np.array([20, 255, 255]) # Extended to 20
        lower_healthy2 = np.array([160, 60, 50]) # Lowered to 160
        upper_healthy2 = np.array([180, 255, 255])
        mask_healthy = cv2.inRange(hsv, lower_healthy1, upper_healthy1) | cv2.inRange(hsv, lower_healthy2, upper_healthy2)
        
    # 3. Fruit Body & Disease Logic (Same as before but using the Adaptive Healthy Mask)
    # Body = Broad Red/Orange/Yellow range + GREEN (if young green fruit)
    # Actually, if we are in Young Phase, Body should include Green.
    lower_body1 = np.array([0, 20, 20])
    upper_body1 = np.array([35, 255, 255]) # Includes Yellow 20-35
    lower_body2 = np.array([155, 20, 20])
    upper_body2 = np.array([180, 255, 255])
    
    mask_body = cv2.inRange(hsv, lower_body1, upper_body1) | cv2.inRange(hsv, lower_body2, upper_body2)
    
    if phase == 'YOUNG':
        # Include Green in Body Mask
        lower_body3 = np.array([30, 20, 20])
        upper_body3 = np.array([90, 255, 255]) 
        mask_body = mask_body | cv2.inRange(hsv, lower_body3, upper_body3)
    
    kernel = np.ones((5,5), np.uint8)
    mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Disease = Body Pixels NOT matched by Healthy Mask
    mask_disease = cv2.bitwise_and(mask_body, cv2.bitwise_not(mask_healthy))
    
    disease_pixels = cv2.countNonZero(mask_disease)
    total_pixels = cv2.countNonZero(mask_body)
    
    if total_pixels == 0:
        return 0.0, "Healthy"
        
    infection_level = (disease_pixels / total_pixels) * 100
    level_val = round(infection_level, 2)
    
    severity = "Healthy"
    if level_val > 50: severity = "High"
    elif level_val > 20: severity = "Moderate"
    elif level_val > 0: severity = "Low"
    
    return level_val, severity

from leaf_filter import is_pomegranate_leaf_candidate, is_pomegranate_fruit_candidate

def main():
    # Load Model
    # Adjust path to where your best.pt ends up
    try:
        model = YOLO('pomegranate_disease_model/yolov8n_cls_run_v2/weights/best.pt')
    except:
        print("Model not found yet. Using yolov8n-cls.pt for testing.")
        model = YOLO('yolov8n-cls.pt')

    # Dual Camera Setup
    caps = []
    # Try opening cameras 0 and 1. 
    for i in range(2):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            caps.append(cap)
        else:
            print(f"Camera {i} could not be opened.")
    
    if not caps:
        print("No cameras found. Exiting.")
        return

    print("Press 'q' to quit.")

    while True:
        for idx, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                continue

            # Preprocessing
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            
            # --- 1. LEAF PIPELINE ---
            lower_green = np.array([25, 40, 40])
            upper_green = np.array([85, 255, 255])
            mask_green = cv2.inRange(hsv, lower_green, upper_green)
            
            # --- 2. FRUIT PIPELINE ---
            # 2. Fruit/Flower Detection (Red/Orange/Yellow/Pink)
            # Red1
            lower_red1 = np.array([0, 20, 20])
            upper_red1 = np.array([10, 255, 255])
            # Red2
            lower_red2 = np.array([160, 20, 20])
            upper_red2 = np.array([180, 255, 255])
            # Orange/Yellow
            lower_orange = np.array([10, 20, 20])
            upper_orange = np.array([35, 255, 255])
            # Pink (Added for Flowers)
            lower_pink = np.array([145, 20, 20])
            upper_pink = np.array([160, 255, 255]) # Overlap slightly with Red2
            
            mask_fruit = cv2.inRange(hsv, lower_red1, upper_red1) | \
                         cv2.inRange(hsv, lower_red2, upper_red2) | \
                         cv2.inRange(hsv, lower_orange, upper_orange) | \
                         cv2.inRange(hsv, lower_pink, upper_pink)
                         
            # Morph Clean
            kernel = np.ones((5,5), np.uint8)
            mask_fruit = cv2.morphologyEx(mask_fruit, cv2.MORPH_OPEN, kernel, iterations=1) # Restored Open=1
            mask_fruit = cv2.morphologyEx(mask_fruit, cv2.MORPH_CLOSE, kernel, iterations=2) # Restored Close=2
            
            mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=3) # Split Canopy (3 iters)
            mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # Find Contours
            contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # ---------------------------------------------------------
            # 1. PROCESS GREEN (Leaves & Green Fruits)
            # ---------------------------------------------------------
            # Green Loop
            for cnt in contours_green:
                area = cv2.contourArea(cnt)
                if area < 1000: continue
                
                # Leaf Check
                if is_pomegranate_leaf_candidate(cnt):
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Just draw 'Leaf' (No Disease Classification on Leaves requested, focus on Fruit)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    # cv2.putText(frame, "Leaf", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # Green Fruit Check (Always Active)
                elif is_pomegranate_fruit_candidate(cnt, strict=True): # Strict for green to avoid leaves
                     x, y, w, h = cv2.boundingRect(cnt)
                     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                     cv2.putText(frame, "Green Fruit", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
            # ---------------------------------------------------------
            # ---------------------------------------------------------
            # 2. PROCESS RED (Fruits)
            # ---------------------------------------------------------
            # Process Red/Orange/Yellow Fruits/Flowers (from mask_fruit)
            contours_fruit, _ = cv2.findContours(mask_fruit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Dynamic Area Threshold (Suppress stems if Giant Fruit exists)
            max_fruit_area = 0
            for cnt in contours_fruit:
                 a = cv2.contourArea(cnt)
                 if a > max_fruit_area: max_fruit_area = a
            
            # Threshold: 500 (Absolute Min) OR 5% of Largest Fruit
            dynamic_min_area = max(500, max_fruit_area * 0.05)
            
            for cnt in contours_fruit:
                a = cv2.contourArea(cnt)
                if a < dynamic_min_area: continue
                
                # Only process flowers if phase is FLOWERING. -> NO, User Removed Phases.
                # Check Flowers (Always)
                if is_flower_candidate(cnt):
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Label Flow
                    if cv2.contourArea(cnt) > 3000:
                        class_name = "Flower Cluster"
                    else:
                        class_name = "Flower"
                        
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
                    cv2.putText(frame, class_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    continue # Skip to next contour if it's a flower and we processed it
            
                # Check if it's a FRUIT (Strict=False for Shapes)
                if is_pomegranate_fruit_candidate(cnt, strict=False):
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w == 0 or h == 0: continue
                    
                    # Potted Plant Fix: Reject Perfect Circles (Ratio > 0.95)
                    # Real Poms have crowns/bumps.
                    ratio = min(w, h) / max(w, h)
                    if ratio > 0.95: continue
                    
                    roi = frame[y:y+h, x:x+w]
                    if roi.size == 0: continue
                    
                    # 1. Disease Classification (YOLOv8 Class)
                    try:
                        results = model(roi, verbose=False)
                        probs = results[0].probs
                        top1_index = probs.top1
                        class_name = model.names[top1_index]
                        confidence = probs.top1conf.item()
                        
                        # Filter Low Confidence (Noise Suppression)
                        # Dynamic Threshold: Large Fruits (>50k px) allow lower confidence (0.25).
                        # Small objects need high confidence (0.45) to avoid noise.
                        # Image 24 Noise was 43% (Area ~15k). Image 25 Giant Fruit was 29% (Area ~640k).
                        min_conf = 0.25 if cv2.contourArea(cnt) > 50000 else 0.45
                        
                        if confidence < min_conf: continue 
                        
                    except:
                        class_name = "Unknown"
                        confidence = 0.0
                    
                    # 2. Infection Level (Color Based)
                    roi_hsv = hsv[y:y+h, x:x+w]
                    # Updated Helper returns Tuple
                    infection_level, severity = get_fruit_infection_level(roi_hsv, None)
                    
                    # Color Logic
                    color = (0, 0, 255) # Red
                    
                    # Draw Red Box (Fruit)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Label construction
                    final_label = f"{class_name} {confidence*100:.0f}%"
                    
                    # Add Severity if detected
                    if infection_level > 0:
                         final_label += f" | Sev:{infection_level}%"
                    
                    cv2.putText(frame, final_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show Feed
            window_name = f"Camera {idx}"
            cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
