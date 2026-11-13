import cv2

import mrcnn.model as modellib
#import dlib

from mrcnn.visualize import InferenceConfig, get_mask_contours, random_colors, draw_mask

#path = r'D:\Eileen_videos\45.mp4'




Xdifference_left = 0

Ydifference_left = 0

Xdifference_right = 0

Ydifference_right = 0

XdifLeft = 0
YdifRight = 0



xLpc= 0
xclrpL= 0

yLpc = 0
yclrpL = 0

xRpc = 0
xclrpR = 0

yRpc = 0
yclrpR = 0



# Pupil center Detection

object_count = 0

# Load Mask-RCNN

config = InferenceConfig(num_classes=1, image_size=512)

model = modellib.MaskRCNN(mode="inference",config=config, model_dir="")

#detector = dlib.get_frontal_face_detector()

model.load_weights(filepath="mask_rcnn_object_0020.h5", by_name=True)



#img = cv2.imread("images/55.jpeg")
img = cv2.imread("4.jpg")
cv2.imshow('img', img)
cv2.waitKey(0)
imag = cv2.resize(img, (512, 512), interpolation= cv2.INTER_LINEAR)
#grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
image = cv2.cvtColor(imag, cv2.COLOR_BGR2RGB)



# Detect objects

r = model.detect([image])[0]

# Finding the center


object_count = len(r["class_ids"])

colors = random_colors(object_count) if object_count > 0 else None
#print(object_count)
for i in range(object_count):
    # 1. Mask
    mask = r["masks"][:, :, i]
    contours = get_mask_contours(mask)
    cnt = contours[0]
    M = cv2.moments(cnt)
    #print (M)
    if i == 0:
      xLpc = int(M['m10']/M['m00'])
      yLpc = int(M['m01']/M['m00'])
      imag = cv2.circle(imag, (xLpc,yLpc), radius=3, color=(0, 0, 255), thickness=-1)
    elif i == 1:
      xRpc = int(M['m10']/M['m00'])
      yRpc = int(M['m01']/M['m00'])
      imag = cv2.circle(imag, (xRpc,yRpc), radius=3, color=(0, 0, 255), thickness=-1)




cv2.imshow('Detection', imag)
cv2.waitKey(0)

#img = cv2.imread("images/55.jpeg")
img = cv2.imread("4.jpg")
#cv2.imshow('img', img)
cv2.waitKey(0)
imag = cv2.resize(img, (512, 512), interpolation= cv2.INTER_LINEAR)
#grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
image = cv2.cvtColor(imag, cv2.COLOR_BGR2RGB)








# CLRP Detection

object_count = 0

# Load Mask-RCNN

config = InferenceConfig(num_classes=1, image_size=512)

model = modellib.MaskRCNN(mode="inference",config=config, model_dir="")

#detector = dlib.get_frontal_face_detector()

model.load_weights(filepath="mask_rcnn_object_0014.h5", by_name=True)



# Detect objects

r = model.detect([image])[0]

# Finding the center


object_count = len(r["class_ids"])

colors = random_colors(object_count) if object_count > 0 else None
#print(object_count)
for i in range(object_count):
    # 1. Mask
    mask = r["masks"][:, :, i]
    contours = get_mask_contours(mask)
    cnt = contours[0]
    M = cv2.moments(cnt)
    #print (M)
    if i == 0:
      xclrpL = int(M['m10']/M['m00'])
      yclrpL = int(M['m01']/M['m00'])
      imag = cv2.circle(imag, (xclrpL,yclrpL), radius=3, color=(0, 255, 0), thickness=-1)
    elif i == 1:
      xclrpR = int(M['m10']/M['m00'])
      yclrpR = int(M['m01']/M['m00'])
      imag = cv2.circle(imag, (xclrpR,yclrpR), radius=3, color=(0, 255, 0), thickness=-1)









cv2.imshow('Detection', imag)
cv2.waitKey(0)


print("xLpc = ",  xLpc)
print("yLpc = ", yLpc)
print("xRpc = ", xRpc)
print("yRpc = ", yRpc)
print("xclrpL = ", xclrpL)
print("yclrpL = ", yclrpL)
print("xclrpR = ", xclrpR)
print("yclrpR = ", yclrpR)








#Strabismus detectinon

Xdifference_left = abs(xLpc - xclrpL)

Ydifference_left = abs(yLpc -yclrpL)

Xdifference_right = abs(xRpc - xclrpR)

Ydifference_right = abs(yRpc - yclrpR)

# Finind

XdifLeft = abs(Xdifference_left - Xdifference_right)
YdifRight = abs(Ydifference_left - Ydifference_right)

print("Xdifference_left = ", Xdifference_left)
print("Ydifference_left = ", Ydifference_left)
print("Xdifference_right = ", Xdifference_right)
print("Ydifference_right = ", Ydifference_right)

print("XdifLeft = ", XdifLeft)
print("YdifRight = ", YdifRight)

#key = cv2.waitKey(1)

#if key == 27:

   # break

# releasing the VideoCapture object

#cap.release()

cv2.destroyAllWindows()