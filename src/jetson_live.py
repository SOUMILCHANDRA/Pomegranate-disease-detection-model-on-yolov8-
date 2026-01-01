import cv2
import time
import sys
from main_analyzer import PomegranateAnalyzer

# Configuration
# Configuration
MODEL_PATH = r"e:\SIH2\pomegranate_disease_model\yolov8n_cls_run_v2\weights\best.pt"  # PC Path
CAMERA_INDEX = 0         # Standard USB Camera
CAMERA_INDEX = 0         # Standard USB Camera (or CSI Camera)

def main():
    print("[Jetson] Initializing Pomegranate AI...")
    try:
        analyzer = PomegranateAnalyzer(MODEL_PATH)
    except Exception as e:
        print(f"[Error] Failed to load Analyzer: {e}")
        return

    print("[Jetson] Opening Camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[Error] Cannot open camera. Check connection.")
        return

    # Set Resolution (Optional - Reduce for speed on Jetson if needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "="*50)
    print("   POMEGRANATE DISEASE SYSTEM - LIVE MODE")
    print("   Press 'S' to SCAN the current plant.")
    print("   Press 'Q' to QUIT.")
    print("   Press 'R' to generate FIELD REPORT (End Session).")
    print("="*50 + "\n")

    last_status = "Ready to Scan"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Error] Failed to capture frame.")
            break

        # Display UI
        display_frame = frame.copy()
        cv2.putText(display_frame, f"STATUS: {last_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press 'S' to SCAN", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Pomegranate Jetson AI", display_frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
            
        elif key == ord('r'):
            print("\n[Action] Generating Field Report...")
            analyzer.generate_field_report()
            last_status = "Report Generated"
            
        else:
            # AUTONOMOUS MODE (Default)
            try:
                success = analyzer.process_frame(frame)
                
                if success:
                    print("\n[Auto] Plant Detected & Scanned!")
                    last_status = "SCANNED! (Paused 5s)"
                    
                    # Visual Feedback
                    cv2.putText(display_frame, "PLANT FOUND & SAVED", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.imshow("Pomegranate Jetson AI", display_frame)
                    cv2.waitKey(1)
                    
                    # Debounce
                    time.sleep(5)
                    last_status = "Ready"
                else:
                    last_status = "Searching..."
                    
            except Exception as e:
                print(f"[Error] Analysis Failed: {e}")
                last_status = "Error"

    cap.release()
    cv2.destroyAllWindows()
    print("[Jetson] System Shutdown.")

if __name__ == "__main__":
    main()
