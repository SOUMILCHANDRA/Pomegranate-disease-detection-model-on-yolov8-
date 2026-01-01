import cv2
import numpy as np

def is_pomegranate_leaf_candidate(contour):
    """
    Filters contours to find Pomegranate Leaves.
    - Shape: Elongated (Oval).
    - Solidity: High (Solid).
    - Excludes: Round objects (Fruits), Irregular (Flowers), Thin lines (Grass).
    """
    area = cv2.contourArea(contour)
    if area < 1000: return False # Increased from 300 to 1000 to ignore small noise/specks
    if area > 100000: return False # Ignore massive walls (Leaves are small)
    
    x, y, w, h = cv2.boundingRect(contour)
    # Aspect Ratio (Width / Height) - Pome leaves are elongated
    # Using MinAreaRect for better orientation handling
    rect = cv2.minAreaRect(contour)
    width = rect[1][0]
    height = rect[1][1]
    if width == 0 or height == 0: return False
    
    ratio = min(width, height) / max(width, height)
    
    # Solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0: return False
    solidity = float(area) / hull_area
    
    # Filter Logic
    # Calibration Result (Dec 11):
    # - Real Pomegranate Leaves (Single) have Solidity > 0.95.
    # - Imposter (Image 24) has Solidity ~0.60 (Serrated/Cluster).
    # Action: Very Strict Solidity, Relaxed Ratio.
    
    threshold_ratio = 0.65    # Allow broader leaves (Real ones were ~0.60-0.68)
    min_ratio = 0.20          # Exclude extremely thin grass
    threshold_solidity = 0.92 # STRICT: Only smooth, single leaves. Kill serrated imposters.
    
    if ratio < threshold_ratio and ratio > min_ratio and solidity > threshold_solidity:
        return True
        
    return False

def is_pomegranate_fruit_candidate(contour, strict=True):
    """
    Filters contours to find Pomegranate Fruits.
    - Shape: Round/Oval but NOT perfect circle/smooth (Rough texture).
    - Solidity: High but not 1.0.
    """
    area = cv2.contourArea(contour)
    if area < 1500: return False # Increased from 1000 to avoid small leaf blobs
    if area > 900000: return False # Increased to 900k for Image 31 (Close-up)
    
    # Aspect Ratio (MinAreaRect)
    rect = cv2.minAreaRect(contour)
    width = rect[1][0]
    height = rect[1][1]
    if width == 0 or height == 0: return False
    
    ratio = min(width, height) / max(width, height)
    
    # Solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0: return False
    solidity = float(area) / hull_area
    
    # Pomegranate Logic:
    # Real Pomegranates are often Knobby/Rough -> Solidity 0.6 - 0.85
    # False Fruits (Citrus/Smooth Leaves) -> Solidity > 0.90
    if solidity > 0.88: return False # Reject too smooth objects
    
    # Real Pomegranates (Young) are often Oval -> Ratio 0.5 - 0.8
    # False Fruits (Perfect Circles) -> Ratio > 0.9
    if ratio > 0.90: return False # Reject perfect circles (unless mature, but mature are usually irregular too)
    
    # strict=True (Flowering/Green Checks)
    # Combined Filter:
    # 1. Round Fruits (Ratio > 0.70): Allow lower solidity (0.65) for rough/giant fruits (e.g., Image 20).
    # 2. Elongated Fruits (Ratio <= 0.70): Require HIGH solidity (0.85) to reject jagged leaf spots (e.g., Image 30).
    
    if strict:
        is_round = (ratio > 0.70)
        
        if is_round:
            if solidity > 0.65: return True
        else:
            # Elongated check
            if ratio > 0.30 and solidity > 0.85: return True
    else:
        # Relaxed (Young/Mature Red Checks)
        # Allow Oval/Rough fruits
        
        # NOTE: Added Strict Ratio Cap to filter Potted Plant aka 'Perfect Circle' (Ratio 0.995)
        # Real Pomegranates are rarely this perfect.
        # ------------------
        # 4. GIANT OBJECT FILTER (> 50k Area)
        # ------------------
        # Hypothesis: Giant Real Fruits are rough/bumpy (Solidity < 0.76).
        # Giant Fake Leaves (Img 25) are smooth (Solidity > 0.77).
        if area > 50000:
            if solidity > 0.76: return False # Reject giant smooth blobs (Leaves/Walls)
    
        # Combined Filter (Smart Geometry):
        # 1. Oval/Round Objects (Ratio > 0.60): Includes Giant/Rough fruits (Image 20, 31).
        #    - Strength: Can be somewhat rough (Solidity > 0.60).
        is_oval_or_round = (ratio > 0.60)
        
        # Advanced Filter Logic
        
        # 0. Global Minimum Size (Reject Noise/Leaves)
        if area < 1500:
            return False

        # 1. Reject Stems (Too thin)
        if ratio < 0.25: 
            return False
            
        # 2. Reject Small Irregular Blobs (Leaves/Noise)
        if area < 2000 and solidity < 0.7:
            return False

        # 3. Acceptance Criteria
        # A. Good Shape (Round/Oval)
        if solidity > 0.7: 
            return True
            
        # B. Large but Irregular (Infected Fruit) - Allow lower solidity if big
        if area > 3000 and solidity > 0.55:
            return True
            
        # C. Round-ish but small
        if ratio > 0.6 and solidity > 0.6:
            return True
            
        return False
            
    return False
