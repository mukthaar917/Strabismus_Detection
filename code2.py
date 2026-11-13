import os
import sys
import numpy as np
import cv2
import types

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import tensorflow.keras.utils as utils

# Explicitly add Mask_RCNN folder to Python system path
mrcnn_path = os.path.abspath(r"D:\Strabismus_Detection\Mask_RCNN")
if mrcnn_path not in sys.path:
    sys.path.insert(0, mrcnn_path)

# ============================================================
# COMPREHENSIVE PATCH FOR MASK R-CNN 
# ============================================================
if not hasattr(keras, 'engine'):
    keras.engine = types.ModuleType('engine')
    keras.engine.Layer = KL.Layer
    keras.engine.InputSpec = KL.InputSpec
    keras.engine.Model = KM.Model

sys.modules['keras.engine'] = keras.engine

topology = types.ModuleType('topology')
topology.get_source_inputs = utils.get_source_inputs
keras.engine.topology = topology
sys.modules['keras.engine.topology'] = topology

tf.random_shuffle = tf.random.shuffle
tf.log = tf.math.log
tf.sets.set_intersection = tf.sets.intersection
tf.sparse_tensor_to_dense = tf.sparse.to_dense
tf.to_float = lambda x: tf.cast(x, tf.float32)

class SafeReshape(tf.keras.layers.Layer):
    def __init__(self, target_shape, **kwargs):
        super().__init__(**kwargs)
        self.target_shape = target_shape
    def call(self, inputs):
        in_shape = tf.shape(inputs)
        batch_size = in_shape[0]
        dyn_shape = [batch_size]
        for dim in self.target_shape:
            if dim is None:
                dyn_shape.append(-1)
            else:
                dyn_shape.append(dim)
        return tf.reshape(inputs, dyn_shape)

KL.Reshape = SafeReshape
keras.layers.Reshape = SafeReshape
tf.keras.layers.Reshape = SafeReshape

# Fix HDF5 weights loading attribute error for TF 2.15+
import h5py
try:
    from keras.saving.legacy import hdf5_format
except ImportError:
    try:
        from tensorflow.python.keras.saving import hdf5_format
    except ImportError:
        try:
            from keras.saving import hdf5_format
        except ImportError:
            import tensorflow.keras.utils as tf_utils
            hdf5_format = tf_utils

saving_module = types.ModuleType('saving')
def patched_load_weights_by_name(f, layers):
    if hasattr(hdf5_format, 'load_weights_from_hdf5_group_by_name'):
        return hdf5_format.load_weights_from_hdf5_group_by_name(f, layers)
    else:
        from tensorflow.python.keras.saving.legacy import hdf5_format as legacy_h5
        return legacy_h5.load_weights_from_hdf5_group_by_name(f, layers)

saving_module.load_weights_from_hdf5_group_by_name = patched_load_weights_by_name
keras.engine.saving = saving_module
sys.modules['keras.engine.saving'] = saving_module

# ============================================================
# IMPORT MRCNN & CONFIG
# ============================================================
import mrcnn.model as modellib
from mrcnn.config import Config
from mrcnn.visualize import random_colors

def patched_refine_detections_graph(rois, probs, deltas, window, config):
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    class_scores = tf.reduce_max(probs, axis=1)
    class_ids_f = tf.cast(class_ids, tf.float32)
    max_inst = config.DETECTION_MAX_INSTANCES
    boxes = tf.zeros([max_inst, 4], dtype=tf.float32)
    class_ids_f = tf.pad(class_ids_f, [[0, tf.maximum(0, max_inst - tf.shape(class_ids_f)[0])]])[:max_inst]
    class_scores = tf.pad(class_scores, [[0, tf.maximum(0, max_inst - tf.shape(class_scores)[0])]])[:max_inst]
    class_ids_col = tf.expand_dims(class_ids_f, axis=1)
    scores_col = tf.expand_dims(class_scores, axis=1)
    dets = tf.concat([boxes, class_ids_col, scores_col], axis=1)
    return dets

modellib.refine_detections_graph = patched_refine_detections_graph

class InferenceConfig(Config):
    NAME = "strabismus_object"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.0  # Allow raw detections through

    def __init__(self, num_classes=1, image_size=512):
        self.NUM_CLASSES = num_classes + 1  
        self.IMAGE_MIN_DIM = image_size
        self.IMAGE_MAX_DIM = image_size
        super().__init__()

def get_mask_contours(mask):
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

# ============================================================
# PATHS & MAIN EXECUTION
# ============================================================
IMAGE_PATH = r"D:\Strabismus_Detection\images\111.png"
OUTPUT_DIR = r"D:\Strabismus_Detection\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
filename = os.path.basename(IMAGE_PATH)

xLpc, yLpc = 0, 0
xRpc, yRpc = 0, 0
xclrpL, yclrpL = 0, 0
xclrpR, yclrpR = 0, 0

img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"Error: Could not load image from {IMAGE_PATH}. Check path.")
    sys.exit()

imag = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
image_rgb = cv2.cvtColor(imag, cv2.COLOR_BGR2RGB)

config = InferenceConfig(num_classes=1, image_size=512)

# ----------------- PUPIL CENTER DETECTION -----------------
pupil_model = modellib.MaskRCNN(mode="inference", config=config, model_dir="")
pupil_model.keras_model.load_weights("mask_rcnn_object_0020.h5", by_name=True)

r_pupil = pupil_model.detect([image_rgb])[0]
pupil_centers = []
for i in range(len(r_pupil["class_ids"])):
    y1, x1, y2, x2 = r_pupil["rois"][i]
    cv2.rectangle(imag, (x1, y1), (x2, y2), (0, 165, 255), 2)  # Orange box
    
    mask = r_pupil["masks"][:, :, i]
    contours = get_mask_contours(mask)
    if len(contours) > 0:
        cnt = contours[0]
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            pupil_centers.append((cx, cy))

pupil_centers = sorted(pupil_centers, key=lambda pt: pt[0])
if len(pupil_centers) >= 2:
    xLpc, yLpc = pupil_centers[0]
    xRpc, yRpc = pupil_centers[1]
    cv2.circle(imag, (xLpc, yLpc), radius=6, color=(0, 0, 255), thickness=-1)     # Bright Red for Left Pupil
    cv2.circle(imag, (xRpc, yRpc), radius=6, color=(255, 0, 255), thickness=-1)  # Purple for Right Pupil

# ------------- CLR DETECTION (WITH INTENSITY FALLBACK) -------------
clr_model = modellib.MaskRCNN(mode="inference", config=config, model_dir="")
clr_model.keras_model.load_weights("mask_rcnn_object_0014.h5", by_name=True)

r_clr = clr_model.detect([image_rgb])[0]
clr_centers = []
for i in range(len(r_clr["class_ids"])):
    mask = r_clr["masks"][:, :, i]
    contours = get_mask_contours(mask)
    if len(contours) > 0:
        cnt = contours[0]
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            clr_centers.append((cx, cy))

# Fallback: Locate brightest point inside pupil regions if model misses diffuse glare
if len(clr_centers) < 2 and len(pupil_centers) >= 2:
    gray_img = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
    clr_centers = []
    for (px, py) in pupil_centers:
        box_size = 15
        x1 = max(0, px - box_size)
        x2 = min(imag.shape[1], px + box_size)
        y1 = max(0, py - box_size)
        y2 = min(imag.shape[0], py + box_size)
        
        roi = gray_img[y1:y2, x1:x2]
        if roi.size > 0:
            _, _, _, max_loc = cv2.minMaxLoc(roi)
            clr_cx = x1 + max_loc[0]
            clr_cy = y1 + max_loc[1]
            clr_centers.append((clr_cx, clr_cy))

clr_centers = sorted(clr_centers, key=lambda pt: pt[0])
if len(clr_centers) >= 2:
    xclrpL, yclrpL = clr_centers[0]
    xclrpR, yclrpR = clr_centers[1]
    cv2.circle(imag, (xclrpL, yclrpL), radius=5, color=(0, 255, 0), thickness=-1)     # Bright Green for Left CLR
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

if Dx > T or Dy > T:
    print("\nStrabismus Detection: STRABISMUS DETECTED[cite: 16]")
    if Dx > Dy + To:
        print("Deviation Type: HORIZONTAL STRABISMUS[cite: 16]")
        
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
        print("Deviation Type: VERTICAL STRABISMUS[cite: 16]")
        print("Subtype: Not classified by this algorithm")
    else:
        print("Deviation Type: UNDETERMINED")
        print("Reason: Dx and Dy differ by no more than To")
else:
    print("\nStrabismus Detection: ORTHOTROPIC / PSEUDO-STRABISMUS")
    print("No strabismus detected using the selected thresholds[cite: 16]")

print("====================================")

output_path = os.path.join(OUTPUT_DIR, f"result_{filename}")
cv2.imwrite(output_path, imag)
print(f"\nAnalysis complete. Result saved to: {output_path}")
