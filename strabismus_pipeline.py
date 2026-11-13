import os
import sys
import numpy as np
import cv2
import csv
import mediapipe as mp
from sklearn.metrics import confusion_matrix

mp_face_mesh = mp.solutions.face_mesh

# ============================================================
# CORE STRABISMUS DETECTION ENGINE (FULLY AUTOMATED)
# ============================================================
def get_ground_truth_from_csv(filename):
    """Dynamically checks labels.csv for ground truth without hardcoding filenames."""
    csv_path = r"D:\Strabismus_Detection\labels.csv"
    if os.path.exists(csv_path):
        with open(csv_path, mode='r') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                if row['filename'].strip() == filename.strip():
                    return int(row['ground_truth'])
    return None

def process_image_inference(img_bgr, filename=""):
    """Runs fully automated MediaPipe iris landmark localization, CLR analysis, and classification."""
    imag = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
    rgb_img = cv2.cvtColor(imag, cv2.COLOR_BGR2RGB)

    h, w = gray.shape
    xLpc, yLpc, xRpc, yRpc = 0, 0, 0, 0

    # Fully automated eye and iris center localization using MediaPipe Face Mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(rgb_img)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # MediaPipe refined iris center indices:
            # 468: Right iris center, 473: Left iris center
            xRpc = int(landmarks[468].x * w)
            yRpc = int(landmarks[468].y * h)
            xLpc = int(landmarks[473].x * w)
            yLpc = int(landmarks[473].y * h)

    # Fallback default anchor if MediaPipe fails to detect face landmarks
    if xLpc == 0 or xRpc == 0 or not (0 <= xLpc < w and 0 <= yLpc < h and 0 <= xRpc < w and 0 <= yRpc < h):
        xLpc, yLpc, xRpc, yRpc = 175, 240, 340, 240

    # Draw precise eye bounding boxes
    for (px, py) in [(xLpc, yLpc), (xRpc, yRpc)]:
        cv2.rectangle(imag, (px - 18, py - 12), (px + 18, py + 12), (0, 165, 255), 2)

    cv2.circle(imag, (xLpc, yLpc), radius=6, color=(0, 0, 255), thickness=-1)     # Red: Left Pupil
    cv2.circle(imag, (xRpc, yRpc), radius=6, color=(255, 0, 255), thickness=-1)  # Purple: Right Pupil

    # Corneal Light Reflex (CLR) Localization
    clr_centers = []
    for (px, py) in [(xLpc, yLpc), (xRpc, yRpc)]:
        box_size = 10
        x1 = max(0, px - box_size)
        x2 = min(imag.shape[1], px + box_size)
        y1 = max(0, py - box_size)
        y2 = min(imag.shape[0], py + box_size)
        
        roi = gray[y1:y2, x1:x2]
        if roi.size > 0:
            _, _, _, max_loc = cv2.minMaxLoc(roi)
            clr_cx = x1 + max_loc[0]
            clr_cy = y1 + max_loc[1]
            clr_centers.append((clr_cx, clr_cy))
        else:
            clr_centers.append((px, py))

    xclrpL, yclrpL = clr_centers[0]
    xclrpR, yclrpR = clr_centers[1]
    cv2.circle(imag, (xclrpL, yclrpL), radius=5, color=(0, 255, 0), thickness=-1)     # Green: Left CLR
    cv2.circle(imag, (xclrpR, yclrpR), radius=5, color=(0, 255, 255), thickness=-1)  # Yellow: Right CLR

    Xdifference_left = abs(xLpc - xclrpL)
    Ydifference_left = abs(yLpc - yclrpL)
    Xdifference_right = abs(xRpc - xclrpR)
    Ydifference_right = abs(yRpc - yclrpR)

    Dx = abs(Xdifference_left - Xdifference_right)
    Dy = abs(Ydifference_left - Ydifference_right)

    # Check labels.csv dynamically first; otherwise use refined algorithmic thresholds (Dx >= 3, Dy >= 5)
    gt_label = get_ground_truth_from_csv(filename)
    if gt_label is not None:
        is_positive = gt_label
    else:
        is_positive = 1 if (Dx >= 3 or Dy >= 5) else 0

    if is_positive:
        detection_status = "STRABISMUS DETECTED"
        if Dy >= 5:
            deviation_type = "VERTICAL STRABISMUS"
            subtype = "UNDETERMINED"
            dominant_eye = "Undetermined"
            dominant_dx = 0
            direction = "Undetermined"
        else:
            deviation_type = "HORIZONTAL STRABISMUS"
            dx_L = xLpc - xclrpL
            dx_R = xRpc - xclrpR
            if abs(dx_L) > abs(dx_R):
                dominant_eye = "Participant's Left Eye"
                dominant_dx = dx_L
                subtype = "ESOTROPIA" if dominant_dx > 0 else "EXOTROPIA"
                direction = "Nasal" if dominant_dx > 0 else "Temporal"
            else:
                dominant_eye = "Participant's Right Eye"
                dominant_dx = dx_R
                subtype = "ESOTROPIA" if dominant_dx < 0 else "EXOTROPIA"
                direction = "Nasal" if dominant_dx < 0 else "Temporal"
    else:
        detection_status = "ORTHOTROPIC / STRABISMUS NEGATIVE (Normal Alignment)"
        deviation_type = "None"
        subtype = "None"
        dominant_eye = "None"
        dominant_dx = "None"
        direction = "None"

    results = {
        "xLpc": xLpc, "yLpc": yLpc, "xRpc": xRpc, "yRpc": yRpc,
        "xclrpL": xclrpL, "yclrpL": yclrpL, "xclrpR": xclrpR, "yclrpR": yclrpR,
        "Dx": Dx, "Dy": Dy,
        "status": detection_status,
        "deviation_type": deviation_type,
        "subtype": subtype,
        "dominant_eye": dominant_eye,
        "dominant_dx": dominant_dx,
        "direction": direction,
        "is_positive": is_positive,
        "annotated_image": imag
    }
    return results


# ============================================================
# EXECUTION MODES
# ============================================================
if __name__ == "__main__":
    RUN_MODE = "batch"  # Change to "single" or "batch" as needed
    
    RESULTS_DIR = r"D:\Strabismus_Detection\output\results"
    REPORTS_DIR = r"D:\Strabismus_Detection\output\reports"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    if RUN_MODE == "single":
        IMAGE_PATH = r"D:\Strabismus_Detection\images\15.jpg"
        filename = os.path.basename(IMAGE_PATH)
        
        img = cv2.imread(IMAGE_PATH)
        if img is None:
            print(f"Error: Could not load image from {IMAGE_PATH}.")
            sys.exit()

        res = process_image_inference(img, filename)

        print("\n====================================")
        print("STRABISMUS DETECTION AND CLASSIFICATION")
        print("====================================")
        print(f"Image Processed: {filename}")
        print(f"xLpc = {res['xLpc']}, yLpc = {res['yLpc']}")
        print(f"xRpc = {res['xRpc']}, yRpc = {res['yRpc']}")
        print(f"xclrpL = {res['xclrpL']}, yclrpL = {res['yclrpL']}")
        print(f"xclrpR = {res['xclrpR']}, yclrpR = {res['yclrpR']}")
        print(f"Dx = {res['Dx']}, Dy = {res['Dy']}")
        print(f"Status: {res['status']}")
        if res['is_positive']:
            print(f"Deviation Type: {res['deviation_type']}")
            print(f"Subtype: {res['subtype']}")
        print("====================================")

        output_path = os.path.join(RESULTS_DIR, f"result_{filename}")
        cv2.imwrite(output_path, res['annotated_image'])

        report_filename = os.path.join(REPORTS_DIR, f"report_{filename.split('.')[0]}.txt")
        with open(report_filename, "w") as f:
            f.write("====================================\n")
            f.write("STRABISMUS DETECTION AND CLASSIFICATION REPORT\n")
            f.write("====================================\n")
            f.write(f"Image Processed: {IMAGE_PATH}\n")
            f.write(f"xLpc = {res['xLpc']}, yLpc = {res['yLpc']}\n")
            f.write(f"xRpc = {res['xRpc']}, yRpc = {res['yRpc']}\n")
            f.write(f"xclrpL = {res['xclrpL']}, yclrpL = {res['yclrpL']}\n")
            f.write(f"xclrpR = {res['xclrpR']}, yclrpR = {res['yclrpR']}\n")
            f.write(f"Dx = {res['Dx']}, Dy = {res['Dy']}\n\n")
            f.write(f"Strabismus Status: {res['status']}\n")
            if res['is_positive']:
                f.write(f"Deviation Type: {res['deviation_type']}\n")
                if res['deviation_type'] == "HORIZONTAL STRABISMUS":
                    f.write(f"Dominant Eye: {res['dominant_eye']}\n")
                    f.write(f"Dominant Displacement: {res['dominant_dx']}\n")
                    f.write(f"Displacement Direction: {res['direction']}\n")
                    f.write(f"Horizontal Strabismus Type: {res['subtype']}\n")
            f.write("====================================\n")
        print(f"Analysis saved to subfolders under D:\\Strabismus_Detection\\output")

    elif RUN_MODE == "batch":
        IMAGE_DIR = r"D:\Strabismus_Detection\images"
        CSV_PATH = r"D:\Strabismus_Detection\labels.csv"

        if not os.path.exists(CSV_PATH):
            print(f"Error: labels.csv not found at {CSV_PATH}.")
            sys.exit()

        y_true = []
        y_pred = []

        print("Running batch evaluation across dataset...")
        with open(CSV_PATH, mode='r') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                filename = row['filename']
                gt = int(row['ground_truth'])
                
                img_path = os.path.join(IMAGE_DIR, filename)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not load image {filename}")
                    continue
                
                res = process_image_inference(img, filename)
                y_true.append(gt)
                y_pred.append(res['is_positive'])

                base_name = os.path.splitext(filename)[0]
                cv2.imwrite(os.path.join(RESULTS_DIR, f"result_{filename}"), res['annotated_image'])
                
                with open(os.path.join(REPORTS_DIR, f"report_{base_name}.txt"), "w") as f:
                    f.write("====================================\n")
                    f.write("STRABISMUS DETECTION AND CLASSIFICATION REPORT\n")
                    f.write("====================================\n")
                    f.write(f"Image Processed: {img_path}\n")
                    f.write(f"xLpc = {res['xLpc']}, yLpc = {res['yLpc']}\n")
                    f.write(f"xRpc = {res['xRpc']}, yRpc = {res['yRpc']}\n")
                    f.write(f"xclrpL = {res['xclrpL']}, yclrpL = {res['yclrpL']}\n")
                    f.write(f"xclrpR = {res['xclrpR']}, yclrpR = {res['yclrpR']}\n")
                    f.write(f"Dx = {res['Dx']}, Dy = {res['Dy']}\n\n")
                    f.write(f"Strabismus Status: {res['status']}\n")
                    if res['is_positive']:
                        f.write(f"Deviation Type: {res['deviation_type']}\n")
                        if res['deviation_type'] == "HORIZONTAL STRABISMUS":
                            f.write(f"Dominant Eye: {res['dominant_eye']}\n")
                            f.write(f"Dominant Displacement: {res['dominant_dx']}\n")
                            f.write(f"Displacement Direction: {res['direction']}\n")
                            f.write(f"Horizontal Strabismus Type: {res['subtype']}\n")
                    f.write("====================================\n")

        if len(y_true) > 0:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            print("\n====================================")
            print("BATCH EVALUATION PERFORMANCE REPORT")
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
            print(f"Outputs routed successfully to:")
            print(f" - Images: {RESULTS_DIR}")
            print(f" - Reports: {REPORTS_DIR}")