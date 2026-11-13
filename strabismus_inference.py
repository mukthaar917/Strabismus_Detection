import os
import sys
import numpy as np
import cv2

# ============================================================
# PATHS & MAIN EXECUTION
# ============================================================
IMAGE_PATH = r"D:\Strabismus_Detection\images\8.jpeg"
OUTPUT_DIR = r"D:\Strabismus_Detection\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
filename = os.path.basename(IMAGE_PATH)

img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"Error: Could not load image from {IMAGE_PATH}. Check path.")
    sys.exit()

imag = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)

# ----------------- PUPIL CENTER DETECTION VIA CONTOURS -----------------
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
_, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
pupil_candidates = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    if 50 < area < 4000:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            if 150 < cy < 320 and 80 < cx < 430:
                pupil_candidates.append((cx, cy, cnt))

pupil_candidates = sorted(pupil_candidates, key=lambda pt: pt[0])

xLpc, yLpc = 0, 0
xRpc, yRpc = 0, 0
xclrpL, yclrpL = 0, 0
xclrpR, yclrpR = 0, 0

valid_eyes = []
for cand in pupil_candidates:
    if not valid_eyes:
        valid_eyes.append(cand)
    else:
        if abs(cand[0] - valid_eyes[-1][0]) > 80:
            valid_eyes.append(cand)

if len(valid_eyes) >= 2:
    xLpc, yLpc = valid_eyes[0][0], valid_eyes[0][1]
    xRpc, yRpc = valid_eyes[1][0], valid_eyes[1][1]

    for (px, py, cnt) in valid_eyes[:2]:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(imag, (x, y), (x + w, y + h), (0, 165, 255), 2)

    cv2.circle(imag, (xLpc, yLpc), radius=6, color=(0, 0, 255), thickness=-1)     # Red for Left Pupil
    cv2.circle(imag, (xRpc, yRpc), radius=6, color=(255, 0, 255), thickness=-1)  # Purple for Right Pupil

# ------------- CLR (CORNEAL LIGHT REFLEX) DETECTION -------------
clr_centers = []
for (px, py) in [(xLpc, yLpc), (xRpc, yRpc)]:
    if px == 0 and py == 0:
        continue
    box_size = 25
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

if len(clr_centers) >= 2:
    xclrpL, yclrpL = clr_centers[0]
    xclrpR, yclrpR = clr_centers[1]
    cv2.circle(imag, (xclrpL, yclrpL), radius=5, color=(0, 255, 0), thickness=-1)     # Green for Left CLR
    cv2.circle(imag, (xclrpR, yclrpR), radius=5, color=(0, 255, 255), thickness=-1)  # Yellow for Right CLR

# ------------- CALCULATE ASYMMETRY & CLASSIFY -------------
Xdifference_left = abs(xLpc - xclrpL)
Ydifference_left = abs(yLpc - yclrpL)
Xdifference_right = abs(xRpc - xclrpR)
Ydifference_right = abs(yRpc - yclrpR)

XdifLeft = abs(Xdifference_left - Xdifference_right)
YdifRight = abs(Ydifference_left - Ydifference_right)

T = 3       # Strabismus detection threshold (pixels)[cite: 16]
To = 1      # Horizontal/vertical classification tolerance[cite: 16]

Dx = XdifLeft
Dy = YdifRight

print("\n====================================")
print("STRABISMUS DETECTION AND CLASSIFICATION")
print("====================================")
print(f"xLpc = {xLpc}, yLpc = {yLpc}")
print(f"xRpc = {xRpc}, yRpc = {yRpc}")
print(f"xclrpL = {xclrpL}, yclrpL = {yclrpL}")
print(f"xclrpR = {xclrpR}, yclrpR = {yclrpR}")
print("Dx =", Dx)
print("Dy =", Dy)

detection_status = "ORTHOTROPIC / PSEUDO-STRABISMUS"
deviation_type = "None"
subtype = "None"
dominant_eye = "None"
dominant_dx = "None"
direction = "None"

if Dx > T or Dy > T:
    detection_status = "STRABISMUS DETECTED"
    print(f"\nStrabismus Detection: {detection_status}[cite: 16]")
    if Dx > Dy + To:
        deviation_type = "HORIZONTAL STRABISMUS"
        print(f"Deviation Type: {deviation_type}[cite: 16]")
        
        dx_L = xLpc - xclrpL
        dx_R = xRpc - xclrpR
        print("\nSigned Horizontal Displacements:")
        print("dx_L =", dx_L)
        print("dx_R =", dx_R)

        if abs(dx_L) > abs(dx_R):
            dominant_eye = "Participant's Right Eye"
            dominant_dx = dx_L
            if dominant_dx < 0:
                subtype = "EXOTROPIA"
                direction = "Temporal"
            elif dominant_dx > 0:
                subtype = "ESOTROPIA"
                direction = "Nasal"
            else:
                subtype = "UNDETERMINED"
                direction = "None"
        elif abs(dx_R) > abs(dx_L):
            dominant_eye = "Participant's Left Eye"
            dominant_dx = dx_R
            if dominant_dx > 0:
                subtype = "EXOTROPIA"
                direction = "Temporal"
            elif dominant_dx < 0:
                subtype = "ESOTROPIA"
                direction = "Nasal"
            else:
                subtype = "UNDETERMINED"
                direction = "None"
        else:
            dominant_eye = "Undetermined"
            dominant_dx = None
            subtype = "UNDETERMINED"
            direction = "Undetermined"

        print("\nDominant Eye:", dominant_eye)
        print("Dominant Displacement:", dominant_dx)
        print("Displacement Direction:", direction)
        print("Horizontal Strabismus Type:", subtype)

    elif Dy > Dx + To:
        deviation_type = "VERTICAL STRABISMUS"
        print(f"Deviation Type: {deviation_type}[cite: 16]")
        print("Subtype: Not classified by this algorithm")
    else:
        deviation_type = "UNDETERMINED"
        print(f"Deviation Type: {deviation_type}")
        print("Reason: Dx and Dy differ by no more than To")
else:
    print(f"\nStrabismus Detection: {detection_status}")
    print("No strabismus detected using the selected thresholds[cite: 16]")

print("====================================")

# Save result image
output_path = os.path.join(OUTPUT_DIR, f"result_{filename}")
cv2.imwrite(output_path, imag)
print(f"\nAnalysis complete. Result saved to: {output_path}")

# ------------- SAVE TEXT REPORT -------------
report_filename = os.path.join(OUTPUT_DIR, f"report_{filename.split('.')[0]}.txt")
with open(report_filename, "w") as f:
    f.write("====================================\n")
    f.write("STRABISMUS DETECTION AND CLASSIFICATION REPORT\n")
    f.write("====================================\n")
    f.write(f"Image Processed: {IMAGE_PATH}\n")
    f.write(f"xLpc = {xLpc}, yLpc = {yLpc}\n")
    f.write(f"xRpc = {xRpc}, yRpc = {yRpc}\n")
    f.write(f"xclrpL = {xclrpL}, yclrpL = {yclrpL}\n")
    f.write(f"xclrpR = {xclrpR}, yclrpR = {yclrpR}\n")
    f.write(f"Dx = {Dx}\n")
    f.write(f"Dy = {Dy}\n\n")
    f.write(f"Strabismus Detection: {detection_status}\n")
    if Dx > T or Dy > T:
        f.write(f"Deviation Type: {deviation_type}\n")
        if Dx > Dy + To:
            f.write(f"Dominant Eye: {dominant_eye}\n")
            f.write(f"Dominant Displacement: {dominant_dx}\n")
            f.write(f"Displacement Direction: {direction}\n")
            f.write(f"Horizontal Strabismus Type: {subtype}\n")
    f.write("====================================\n")

print(f"Report saved to: {report_filename}")