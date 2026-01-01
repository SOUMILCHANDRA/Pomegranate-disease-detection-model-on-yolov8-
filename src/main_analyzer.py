import cv2
import numpy as np
import sys
from ultralytics import YOLO

# Import Filters (Reusing proven logic)
try:
    from leaf_filter import is_pomegranate_leaf_candidate, is_pomegranate_fruit_candidate
except ImportError:
    print("Error: leaf_filter.py not found. Please ensure it is in the same directory.")
    sys.exit(1)

# Helper: Fruit Infection Level Calculation (Copied/Refined for Analyzer)
def calculate_disease_severity(fruit_roi_hsv):
    """
    Calculates disease severity percentage based on Black/Brown/Yellow spots on the fruit.
    Input: ROI in HSV format.
    Output: Percentage (0.0 to 100.0).
    """
    total_pixels = fruit_roi_hsv.shape[0] * fruit_roi_hsv.shape[1]
    if total_pixels == 0: return 0.0

    # Disease Colors:
    # 1. Black/Dark Brown (Anthracnose/Bacterial Blight spots)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60]) # Very dark

    # 2. Brown/Rot (General Disease) - H 0-30, S > 30, V 20-150
    lower_brown = np.array([0, 30, 20])
    upper_brown = np.array([30, 255, 150])
    
    # 3. Yellow Halo (Bacterial Blight Halo) - H 20-35
    lower_yellow = np.array([20, 40, 40])
    upper_yellow = np.array([35, 255, 255])
    
    mask_black = cv2.inRange(fruit_roi_hsv, lower_black, upper_black)
    mask_brown = cv2.inRange(fruit_roi_hsv, lower_brown, upper_brown)
    mask_yellow = cv2.inRange(fruit_roi_hsv, lower_yellow, upper_yellow)
    
    # Combine masks
    disease_mask = cv2.bitwise_or(mask_black, mask_brown)
    disease_mask = cv2.bitwise_or(disease_mask, mask_yellow)
    
    disease_pixels = cv2.countNonZero(disease_mask)
    
    return round((disease_pixels / total_pixels) * 100, 2)


class PomegranateAnalyzer:
    def __init__(self, model_path):
        self.model_path = model_path
        print(f"Loading Model from {model_path}...")
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None
            
        # Batch Session State
        self.batch_stats = {
            "TotalImages": 0,
            "Values": {} # Key: DiseaseName, Value: List of Severities
        }

    def process_image(self, img_path):
        """
        Reads image from path and processes it.
        """
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Error: Could not read image {img_path}")
            return
            
        print(f"\nProcessing: {img_path}")
        self.process_frame(frame)
        
    def process_frame(self, frame):
        """
        Main Pipeline for a single frame (Image or Video Frame).
        1. Validate Plant
        2. Detect Fruits
        3. Analyze Disease
        4. Accumulate Stats
        """
        self.batch_stats["TotalImages"] += 1
        
        # Preprocessing
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # ---------------------------------------------------
        # STEP 1: VALIDATE PLANT (Is it Pomegranate?)
        # ---------------------------------------------------
        is_pome_leaf, leaf_count = self.validate_plant(hsv)
        
        # ---------------------------------------------------
        # STEP 2: DETECT FRUITS
        # ---------------------------------------------------
        fruit_contours, max_raw_area = self.detect_fruits(hsv)
        fruit_candidate_count = len(fruit_contours)
        
        # Validation Logic:
        if not is_pome_leaf and fruit_candidate_count == 0:
             # print(f"Result: NOT A POMEGRANATE PLANT (Leaves: {leaf_count}, Potential Fruits: 0)")
             # In live mode, we don't need to save "Not Plant" images constantly.
             return False
             
        # print(f"Plant Validated. Leaves: {leaf_count}, Potential Fruits: {fruit_candidate_count}")
        
        if fruit_candidate_count == 0:
            print("Result: Pomegranate Plant Verified (No Fruits Detected)")
            cv2.putText(frame, "Pomegranate Plant (No Fruits)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite("result_analyzer.jpg", frame)
            return True

        # ---------------------------------------------------
        # STEP 3: ANALYZE DISEASE
        # ---------------------------------------------------
        detected_count = 0
        
        # Dynamic Area Prep
        dynamic_min_area = max(500, max_raw_area * 0.05)
        
        for cnt in fruit_contours:
            area = cv2.contourArea(cnt)
            if area < dynamic_min_area: continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0 or h == 0: continue
            ratio = min(w, h) / max(w, h)
            
            # Strict Ratio Cap
            if ratio > 0.95: continue 

            # Shape Filter
            if not is_pomegranate_fruit_candidate(cnt, strict=False):
                continue
                
            detected_count += 1
            
            # ROI Extraction
            roi = frame[y:y+h, x:x+w]
            roi_hsv = hsv[y:y+h, x:x+w]
            
            # 3a. Classification (YOLO)
            label = "Fruit"
            conf = 0.0
            class_name = "Healthy" # Default
            
            if self.model:
                try:
                    results = self.model(roi, verbose=False)
                    probs = results[0].probs
                    top1_index = probs.top1
                    class_name = self.model.names[top1_index]
                    conf = probs.top1conf.item()
                    
                    # 3b. Confidence Filter
                    min_conf = 0.25 if area > 50000 else 0.45
                    
                    if conf < min_conf:
                        detected_count -= 1
                        continue 
                        
                    label = f"{class_name} {conf*100:.0f}%"
                except Exception as e:
                    print(f"Model Error: {e}")
            
            # 3c. Infection Severity
            severity = calculate_disease_severity(roi_hsv)
            if severity > 0:
                label += f" | Inf:{severity}%"
                
            print(f"  -> Detected: {label}")
            
            # Batch Data Collection
            if class_name not in self.batch_stats["Values"]:
                self.batch_stats["Values"][class_name] = []
            self.batch_stats["Values"][class_name].append(severity)

            # 3d. Treatment Recommendation
            TREATMENTS = {
                "Anthracnose": [
                     {"Chemical": "Propulse (170-480 g/acre)", "Company": "Bayer", "Min": 170, "Max": 480, "Unit": "g"},
                     {"Chemical": "Score (1 L/acre)", "Company": "Syngenta", "Min": 1000, "Max": 1000, "Unit": "mL"}
                ],
                "Bacterial_Blight": [
                     {"Chemical": "Streptocycline (500ppm) + Copper Oxy (0.2%)", "Company": "Bayer", "Min": 2000, "Max": 2000, "Unit": "g"},
                     {"Chemical": "Kasugamycin 5% + Copper Oxy 45%", "Company": "Syngenta", "Min": 1000, "Max": 1000, "Unit": "g"} 
                ],
                "Cercospora": [
                     {"Chemical": "Propiconazole / Difenoconazole", "Company": "Bayer", "Min": 200, "Max": 200, "Unit": "g"},
                     {"Chemical": "Mancozeb (2.5kg)", "Company": "Syngenta", "Min": 2500, "Max": 2500, "Unit": "g"}
                ],
                "Alternaria": [
                     {"Chemical": "Luna Experience (150-200 mL/acre)", "Company": "Bayer", "Min": 150, "Max": 200, "Unit": "mL"},
                     {"Chemical": "Amistar (200-250 mL/acre)", "Company": "Syngenta", "Min": 200, "Max": 250, "Unit": "mL"}
                ]
            }
            
            # Print to Console only (Simplify visual output to avoid clutter)
            if class_name in TREATMENTS and class_name != "Healthy":
                pass 
                
            # Draw
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Simplified Text on Image
            if class_name in TREATMENTS:
                 rec_text = f"Rec: {TREATMENTS[class_name][0]['Chemical']}"
                 cv2.putText(frame, rec_text, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if detected_count == 0:
             print("Result: Potential Fruits filtered out (False Positives). Final: No Fruits.")
             
             # RETROACTIVE VALIDATION:
             if leaf_count < 3:
                 print(f"REVOKING STATUS: Leaves ({leaf_count}) < 3 and Fruits (0). NOT A POMEGRANATE PLANT.")
                 cv2.putText(frame, "Not Pomegranate Plant", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                 cv2.imwrite("result_analyzer.jpg", frame)
                 return False

             cv2.putText(frame, "Pomegranate Plant (No Fruits)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
             print(f"Result: {detected_count} Fruits Analyzed.")
        
        cv2.imwrite("result_analyzer.jpg", frame)
        return True
        
    def generate_field_report(self):
        print("\n" + "="*50)
        print("       FIELD PRESCRIPTION REPORT")
        print("="*50)
        print(f"Total Plants Scanned: {self.batch_stats['TotalImages']}")
        
        # Prepare Data for Firebase
        firebase_data = {
            "total_plants": self.batch_stats['TotalImages'],
            "status": "Healthy",
            "detected_diseases": []
        }
        
        if not self.batch_stats["Values"]:
            print("No diseases detected. Field is Healthy.")
            firebase_data["status"] = "Healthy"
        else:
            firebase_data["status"] = "Infected"
            
            # Pricing/Efficacy Data (Source: Manual Extraction)
            TREATMENTS = {
                "Anthracnose": [
                    {"Chemical": "Propulse", "Company": "Bayer", "Min": 170, "Max": 480, "Unit": "g", "Price": 3100},
                    {"Chemical": "Score",    "Company": "Syngenta", "Min": 1000, "Max": 1000, "Unit": "mL", "Price": 1250}
                ],
                "Bacterial_Blight": [
                    {"Chemical": "Streptocycline+Copper", "Company": "Bayer", "Min": 2000, "Max": 2000, "Unit": "g", "Price": 950},
                    {"Chemical": "Kasugamycin+Copper",    "Company": "Syngenta", "Min": 1000, "Max": 1000, "Unit": "g", "Price": 1300}
                ],
                "Cercospora": [
                    {"Chemical": "Propiconazole", "Company": "Bayer", "Min": 200, "Max": 250, "Unit": "g", "Price": 1750},
                    {"Chemical": "Mancozeb",      "Company": "Syngenta", "Min": 2500, "Max": 2500, "Unit": "g", "Price": 1050}
                ],
                "Alternaria": [
                    {"Chemical": "Luna Experience", "Company": "Bayer", "Min": 150, "Max": 200, "Unit": "mL", "Price": 3800},
                    {"Chemical": "Amistar/Score",   "Company": "Syngenta", "Min": 200, "Max": 250, "Unit": "mL", "Price": 2300}
                ]
            }
            
            for disease, severities in self.batch_stats["Values"].items():
                if disease == "Healthy": continue
                
                mean_severity = sum(severities) / len(severities)
                max_severity = max(severities)
                print(f"\nDISEASE: {disease}")
                print(f"  - Detected in {len(severities)} fruits.")
                print(f"  - Average Infection: {mean_severity:.2f}%")
                
                disease_entry = {
                    "disease_name": disease,
                    "avg_infection": float(f"{mean_severity:.2f}"),
                    "prescriptions": []
                }
                
                # DOSAGE CALCULATOR
                print("  -> RECOMMENDED PRESCRIPTION (Per Acre):")
                
                if disease in TREATMENTS:
                    for opt in TREATMENTS[disease]:
                        # Calculation
                        base = opt["Min"]
                        ceiling = opt["Max"]
                        factor = min(1.0, mean_severity / 80.0)
                        calc_dosage = base + (ceiling - base) * factor
                        
                        print(f"    * {opt['Company']} {opt['Chemical']}: {calc_dosage:.0f} {opt['Unit']} / acre")
                        print(f"      (Cost: {opt['Price']} RS)")
                        
                        disease_entry["prescriptions"].append({
                            "company": opt["Company"],
                            "chemical": opt["Chemical"],
                            "dosage": float(f"{calc_dosage:.0f}"),
                            "unit": opt["Unit"],
                            "price_rs": opt["Price"]
                        })
                
                firebase_data["detected_diseases"].append(disease_entry)
        
        print("="*50 + "\n")
        
        # Initialize Firebase (Cloud)
        try:
            from firebase_connector import FirebaseConnector
            connector_fb = FirebaseConnector()
            connector_fb.push_report(firebase_data)
        except ImportError:
            print("[Warning] 'firebase-admin' not installed. Skipping Cloud Upload.")
        except Exception as e:
            print(f"[Firebase] Integration Error: {e}")

        # Initialize Postgres (Local/Backend)
        try:
            from postgres_connector import PostgresConnector
            connector_pg = PostgresConnector()
            # Since the data structure for Postgres might slightly differ or be reusable, 
            # we check if we can reuse 'firebase_data' which matches the structure required by save_report
            # (total_plants, status, detected_diseases -> [disease_name, avg_infection, prescriptions])
            # It matches perfectly.
            connector_pg.save_report(firebase_data)
        except ImportError:
            print("[Warning] 'sqlalchemy' not installed. Skipping DB Save.")
        except Exception as e:
            print(f"[Postgres] Integration Error: {e}")

    def validate_plant(self, hsv):
        """
        Step 1: Check for Pomegranate Leaves.
        Returns: (is_pomegranate: bool, leaf_count: int)
        """
        # Green Mask
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Cleanup
        kernel = np.ones((5,5), np.uint8)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=2)
        
        contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_leaves = 0
        for cnt in contours:
            if is_pomegranate_leaf_candidate(cnt):
                valid_leaves += 1
                
        # Threshold: Need at least 3 leaves to call it a "Plant"
        # Or if usage is extremely zoomed out, maybe area?
        # Let's say 3 leaves is a safe bet for a "Plant" photo.
        return (valid_leaves >= 3, valid_leaves)

    def detect_fruits(self, hsv):
        """
        Step 2: Check for Red/Orange/Pink Blobs.
        Returns: List of contours.
        """
        # Color Ranges (Wide net, filter later)
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
        
        # Calculate Max Raw Area (for Dynamic Thresholding in Main Loop)
        max_raw_area = 0
        for cnt in contours:
            a = cv2.contourArea(cnt)
            if a > max_raw_area: max_raw_area = a
        
        # Filter Contours by Shape immediately
        valid_fruits = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000: continue # Increased min size to 1000
            
            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0 or h == 0: continue
            ratio = min(w, h) / max(w, h)
            
            # Global Filters
            if ratio > 0.95: continue # Reject Circles
            
            # Shape Filter
            if is_pomegranate_fruit_candidate(cnt, strict=False):
                valid_fruits.append(cnt)
                
        return valid_fruits, max_raw_area

if __name__ == "__main__":
    import os
    # Get the project root directory (assuming this script is in src/)
    # src/main_analyzer.py -> .. -> project_root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    model_path = os.path.join(project_root, "model", "pomegranate_disease_model", "yolov8n_cls_run_v2", "weights", "best.pt")
    
    analyzer = PomegranateAnalyzer(model_path)
    
    # Test Batch
    test_image_path = os.path.join(project_root, "data", "test_samples", "test_image_37.jpg")
    test_images = [test_image_path]
    
    for img in test_images:
        analyzer.process_image(img)
        
    # Generate Final Report
    analyzer.generate_field_report()
