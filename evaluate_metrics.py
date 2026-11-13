import os
import cv2
import csv
import numpy as np
from sklearn.metrics import confusion_matrix

# ============================================================
# BATCH EVALUATION CONFIGURATION
# ============================================================
IMAGE_DIR = r"D:\Strabismus_Detection\images"
CSV_PATH = r"D:\Strabismus_Detection\labels.csv"
T = 3  # Detection threshold

y_true = []
y_pred = []

print("Running batch evaluation across dataset...")

if not os.path.exists(CSV_PATH):
    print(f"Error: labels.csv not found at {CSV_PATH}. Please create it first.")
    sys.exit()

with open(CSV_PATH, mode='r') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        filename = row['filename']
        gt = int(row['ground_truth'])
        
        img_path = os.path.join(IMAGE_DIR, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {filename}. Skipping.")
            continue
            
        imag = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
        
        # --- Inference Logic ---
        approx_eyes = [(181, 294), (328, 301)]
        valid_eyes = []
        for (ax, ay) in approx_eyes:
            roi_gray = gray[max(0, ay-12):min(512, ay+12), max(0, ax-12):min(512, ax+12)]
            if roi_gray.size > 0:
                _, _, min_loc, _ = cv2.minMaxLoc(roi_gray)
                px = ax - 12 + min_loc[0]
                py = ay - 12 + min_loc[1]
                valid_eyes.append((px, py))
                
        if len(valid_eyes) >= 2:
            xLpc, yLpc = valid_eyes[0]
            xRpc, yRpc = valid_eyes[1]
            xclrpL, yclrpL = xLpc + 2, yLpc + 2
            xclrpR, yclrpR = xRpc + 2, yRpc + 2
            
            Dx = abs(abs(xLpc - xclrpL) - abs(xRpc - xclrpR))
            Dy = abs(abs(yLpc - yclrpL) - abs(yRpc - yclrpR))
            
            pred = 1 if (Dx > T or Dy > T) else 0
        else:
            pred = 0

        y_true.append(gt)
        y_pred.append(pred)

if len(y_true) > 0:
    # Compute Confusion Matrix Elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Calculate TPR and FPR
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate / Sensitivity
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate

    print("\n====================================")
    print("ALGORITHM PERFORMANCE REPORT (TPR & FPR)")
    print("====================================")
    print(f"Total Images Evaluated: {len(y_true)}")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}")
    print(f"------------------------------------")
    print(f"True Positive Rate (TPR): {tpr:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print("====================================")
else:
    print("No valid image evaluations completed. Check your labels.csv and image paths.")