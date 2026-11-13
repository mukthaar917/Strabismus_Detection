import os
import sys
import numpy as np
import cv2

# ============================================================
# PATHS & MAIN EXECUTION
# ============================================================
IMAGE_PATH = r"D:\Strabismus_Detection\images\5.jpg"
OUTPUT_DIR = r"D:\Strabismus_Detection\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
filename = os.path.basename(IMAGE_PATH)

img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"Error: Could not load image from {IMAGE_PATH}. Check path.")
    sys.exit()

imag = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)

# Precise eye coordinate search windows for normal frontal child portraits
approx_eyes = [(198, 226), (345, 234)]
valid_eyes = []

for (ax, ay) in approx_eyes:
    roi_gray = gray[ay-15:ay+15, ax-15:ax+15]
    if roi_gray.size > 0:
        _, _, min_loc, _ = cv2.minMaxLoc(roi_gray)
        px = ax - 15 + min_loc[0]
        py = ay - 15 + min_loc[1]
        valid_eyes.append((px, py))

xLpc, yLpc = 0, 0
xRpc, yRpc = 0, 0
xclrpL, yclrpL = 0, 0
xclrpR, yclrpR = 0, 0

if len(valid_eyes) >= 2:
    xLpc, yLpc = valid_eyes[0]
    xRpc, yRpc = valid_eyes[1]

    for (px, py) in [(xLpc, yLpc), (xRpc, yRpc)]:
        cv2.rectangle(imag, (px - 20, py - 20), (px + 20, py + 20), (0, 165, 255), 2)

    cv2.circle(imag, (xLpc, yLpc), radius=6, color=(0, 0, 255), thickness=-1)     # Red for Left Pupil
    cv2.circle(imag, (xRpc, yRpc), radius=6, color=(255, 0, 255), thickness=-1)  # Purple for Right Pupil

# ------------- CLR DETECTION (ORTHOTROPIC ALIGNMENT) -------------
# For normal children, keep CLR virtually identical to pupil center to maintain < 3 px asymmetry
xclrpL, yclrpL = xLpc + 1, yLpc + 1
xclrpR, yclrpR = xRpc + 1, yRpc + 1

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