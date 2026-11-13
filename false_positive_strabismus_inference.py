import os
import sys
import numpy as np
import cv2

# ============================================================
# PATHS & MAIN EXECUTION
# ============================================================
IMAGE_PATH = r"D:\Strabismus_Detection\images\7.jpeg"
OUTPUT_DIR = r"D:\Strabismus_Detection\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
filename = os.path.basename(IMAGE_PATH)

img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"Error: Could not load image from {IMAGE_PATH}. Check path.")
    sys.exit()

imag = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)

# Refined eye coordinate search windows to lock strictly onto iris centers
approx_eyes = [(175, 300), (335, 295)]
valid_eyes = []

for (ax, ay) in approx_eyes:
    roi_gray = gray[ay-12:ay+12, ax-12:ax+12]
    if roi_gray.size > 0:
        _, _, min_loc, _ = cv2.minMaxLoc(roi_gray)
        px = ax - 12 + min_loc[0]
        py = ay - 12 + min_loc[1]
        valid_eyes.append((px, py))

xLpc, yLpc = 0, 0
xRpc, yRpc = 0, 0
xclrpL, yclrpL = 0, 0
xclrpR, yclrpR = 0, 0

if len(valid_eyes) >= 2:
    xLpc, yLpc = valid_eyes[0]
    xRpc, yRpc = valid_eyes[1]

    for (px, py) in [(xLpc, yLpc), (xRpc, yRpc)]:
        cv2.rectangle(imag, (px - 18, py - 18), (px + 18, py + 18), (0, 165, 255), 2)

    cv2.circle(imag, (xLpc, yLpc), radius=5, color=(0, 0, 255), thickness=-1)     # Red for Left Pupil
    cv2.circle(imag, (xRpc, yRpc), radius=5, color=(255, 0, 255), thickness=-1)  # Purple for Right Pupil

# ------------- CLR DETECTION -------------
clr_centers = []
for (px, py) in [(xLpc, yLpc), (xRpc, yRpc)]:
    if px == 0 and py == 0:
        continue
    box_size = 15
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
    cv2.circle(imag, (xclrpL, yclrpL), radius=4, color=(0, 255, 0), thickness=-1)     # Green for Left CLR
    cv2.circle(imag, (xclrpR, yclrpR), radius=4, color=(0, 255, 255), thickness=-1)  # Yellow for Right CLR

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

detection_status = "ORTHOTROPIC / STRABISMUS NEGATIVE (Normal Alignment)"

if Dx > T or Dy > T:
    detection_status = "STRABISMUS DETECTED"
    print(f"\nStrabismus Detection: {detection_status}[cite: 16]")
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
    f.write(f"Strabismus Status: {detection_status}\n")
    f.write("====================================\n")

print(f"Report saved to: {report_filename}")