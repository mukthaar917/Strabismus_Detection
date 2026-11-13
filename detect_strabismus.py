import cv2
import mrcnn.model as modellib
from mrcnn.visualize import InferenceConfig, get_mask_contours

# Set the image path based on your workspace folder structure
# Change this to "images/112.png", "images/168.png", etc., to test other files.
IMAGE_PATH = "images/111.png" 

# Initialize coordinate variables
xLpc, yLpc = 0, 0
xRpc, yRpc = 0, 0
xclrpL, yclrpL = 0, 0
xclrpR, yclrpR = 0, 0

# Load image
img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"Error: Could not load image from {IMAGE_PATH}. Check the file path.")
    exit()

cv2.imshow('Original Image', img)
cv2.waitKey(0)

# Resize and convert colors for Mask R-CNN
imag = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
image_rgb = cv2.cvtColor(imag, cv2.COLOR_BGR2RGB)

# ============================================================
# PUPIL CENTER DETECTION
# ============================================================
config = InferenceConfig(num_classes=1, image_size=512)
pupil_model = modellib.MaskRCNN(mode="inference", config=config, model_dir="")
pupil_model.load_weights(filepath="mask_rcnn_object_0020.h5", by_name=True)

r_pupil = pupil_model.detect([image_rgb])[0]
object_count = len(r_pupil["class_ids"])

for i in range(object_count):
    mask = r_pupil["masks"][:, :, i]
    contours = get_mask_contours(mask)
    if len(contours) > 0:
        cnt = contours[0]
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            if i == 0:
                xLpc = int(M['m10']/M['m00'])
                yLpc = int(M['m01']/M['m00'])
                cv2.circle(imag, (xLpc, yLpc), radius=3, color=(0, 0, 255), thickness=-1)
            elif i == 1:
                xRpc = int(M['m10']/M['m00'])
                yRpc = int(M['m01']/M['m00'])
                cv2.circle(imag, (xRpc, yRpc), radius=3, color=(0, 0, 255), thickness=-1)

cv2.imshow('Pupil Detection', imag)
cv2.waitKey(0)

# ============================================================
# CLR (CORNEAL LIGHT REFLEX) DETECTION
# ============================================================
clr_model = modellib.MaskRCNN(mode="inference", config=config, model_dir="")
clr_model.load_weights(filepath="mask_rcnn_object_0014.h5", by_name=True)

r_clr = clr_model.detect([image_rgb])[0]
object_count = len(r_clr["class_ids"])

for i in range(object_count):
    mask = r_clr["masks"][:, :, i]
    contours = get_mask_contours(mask)
    if len(contours) > 0:
        cnt = contours[0]
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            if i == 0:
                xclrpL = int(M['m10']/M['m00'])
                yclrpL = int(M['m01']/M['m00'])
                cv2.circle(imag, (xclrpL, yclrpL), radius=3, color=(0, 255, 0), thickness=-1)
            elif i == 1:
                xclrpR = int(M['m10']/M['m00'])
                yclrpR = int(M['m01']/M['m00'])
                cv2.circle(imag, (xclrpR, yclrpR), radius=3, color=(0, 255, 0), thickness=-1)

cv2.imshow('Pupil & CLR Detection', imag)
cv2.waitKey(0)

# Print coordinates
print(f"xLpc = {xLpc}, yLpc = {yLpc}")
print(f"xRpc = {xRpc}, yRpc = {yRpc}")
print(f"xclrpL = {xclrpL}, yclrpL = {yclrpL}")
print(f"xclrpR = {xclrpR}, yclrpR = {yclrpR}")

# ============================================================
# STRABISMUS DETECTION AND CLASSIFICATION
# ============================================================
Xdifference_left = abs(xLpc - xclrpL)
Ydifference_left = abs(yLpc - yclrpL)
Xdifference_right = abs(xRpc - xclrpR)
Ydifference_right = abs(yRpc - yclrpR)

Dx = abs(Xdifference_left - Xdifference_right)
Dy = abs(Ydifference_left - Ydifference_right)

print("\n====================================")
print("STRABISMUS DETECTION AND CLASSIFICATION")
print("====================================")
print("Dx =", Dx)
print("Dy =", Dy)

T = 3       # Strabismus detection threshold (pixels)
To = 1      # Horizontal/vertical classification tolerance

if Dx > T or Dy > T:
    print("\nStrabismus Detection: STRABISMUS DETECTED")
    if Dx > Dy + To:
        print("Deviation Type: HORIZONTAL STRABISMUS")
        dx_L = xLpc - xclrpL
        dx_R = xRpc - xclrpR
        print(f"\nSigned Horizontal Displacements:\ndx_L = {dx_L}\ndx_R = {dx_R}")

        if abs(dx_L) > abs(dx_R):
            dominant_eye = "Participant's Right Eye"
            dominant_dx = dx_L
            if dominant_dx < 0:
                subtype, direction = "EXOTROPIA", "Temporal"
            elif dominant_dx > 0:
                subtype, direction = "ESOTROPIA", "Nasal"
            else:
                subtype, direction = "UNDETERMINED", "None"
        elif abs(dx_R) > abs(dx_L):
            dominant_eye = "Participant's Left Eye"
            dominant_dx = dx_R
            if dominant_dx > 0:
                subtype, direction = "EXOTROPIA", "Temporal"
            elif dominant_dx < 0:
                subtype, direction = "ESOTROPIA", "Nasal"
            else:
                subtype, direction = "UNDETERMINED", "None"
        else:
            dominant_eye, dominant_dx, subtype, direction = "Undetermined", None, "UNDETERMINED", "Undetermined"

        print(f"\nDominant Eye: {dominant_eye}\nDominant Displacement: {dominant_dx}\nDisplacement Direction: {direction}\nHorizontal Strabismus Type: {subtype}")
    elif Dy > Dx + To:
        print("Deviation Type: VERTICAL STRABISMUS\nSubtype: Not classified by this algorithm")
    else:
        print("Deviation Type: UNDETERMINED\nReason: Dx and Dy differ by no more than To")
else:
    print("\nStrabismus Detection: ORTHOTROPIC\nNo strabismus detected using the selected thresholds")

print("\n====================================\nANALYSIS COMPLETED\n====================================")
cv2.destroyAllWindows()