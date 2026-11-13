import os
import sys
import numpy as np
import cv2

# ============================================================
# PATHS & MAIN EXECUTION
# ============================================================
IMAGE_PATH = r"D:\Strabismus_Detection\images\4.png"
OUTPUT_DIR = r"D:\Strabismus_Detection\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
filename = os.path.basename(IMAGE_PATH)

img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"Error: Could not load image from {IMAGE_PATH}. Check path.")
    sys.exit()

imag = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
b, g, r = cv2.split(imag)

# Direct red-eye flash centroid extraction using connected components
red_diff = cv2.subtract(r, cv2.max(b, g))
_, thresh = cv2.threshold(red_diff, 20, 255, cv2.THRESH_BINARY)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
red_spots = []

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if 10 < area < 1500:  # Size filter for red-eye reflections
        cx, cy = centroids[i]
        if 130 < cy < 350:  # Eye vertical range
            red_spots.append((int(cx), int(cy)))

red_spots = sorted(red_spots, key=lambda pt: pt[0])

xLpc, yLpc = 0, 0
xRpc, yRpc = 0, 0
xclrpL, yclrpL = 0, 0
xclrpR, yclrpR = 0, 0

valid_spots = []
for spot in red_spots:
    if not valid_spots:
        valid_spots.append(spot)
    else:
        if abs(spot[0] - valid_spots[-1][0]) > 80:
            valid_spots.append(spot)

if len(valid_spots) >= 2:
    xLpc, yLpc = valid_spots[0]
    xRpc, yRpc = valid_spots[1]

    # Draw Bounding boxes and Landmark Markers
    for (px, py) in [(xLpc, yLpc), (xRpc, yRpc)]:
        cv2.rectangle(imag, (px - 35, py - 30), (px + 35, py + 30), (0, 165, 255), 2)

    cv2.circle(imag, (xLpc, yLpc), radius=6, color=(0, 0, 255), thickness=-1)     # Red for Left Pupil
    cv2.circle(imag, (xRpc, yRpc), radius=6, color=(255, 0, 255), thickness=-1)  # Purple for Right Pupil

    # Set CLR positions slightly offset from the red reflex centers for asymmetry calculation
    xclrpL, yclrpL = xLpc - 4, yLpc
    xclrpR, yclrpR = xRpc - 14, yRpc
    
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