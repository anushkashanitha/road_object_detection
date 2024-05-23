#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
import numpy as np
import cv2
import streamlit as st
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # You can use different model sizes like 'yolov8s.pt', 'yolov8m.pt', etc.


st.title('Upload and Process .npz File')

uploaded_file1 = st.file_uploader("Choose a base.npz file", type="npz")

if uploaded_file is not None:
    base_data = np.load(uploaded_file1)
else:
    uploaded_file1 = st.file_uploader("Choose a base.npz file", type="npz")

st.title('Upload and Process .npz File')

uploaded_file2 = st.file_uploader("Choose a test.npz file", type="npz")

if uploaded_file is not None:
    test_data = np.load(uploaded_file2)
else:
    uploaded_file2 = st.file_uploader("Choose a test.npz file", type="npz")
    

# Load data from .npz files
#base_data = np.load('base.npz')
#test_data = np.load('test.npz')

base_images = base_data["images"]
base_gps = base_data["gps"]
base_compass = base_data["compass"]

test_images = test_data["images"]
test_gps = test_data["gps"]
test_compass = test_data["compass"]

canvas = np.zeros((800, 800, 3))

# Initialize background subtractor
back_sub = cv2.createBackgroundSubtractorMOG2()

# Function to detect objects using YOLOv8
def detect_objects_yolo(image):
    results = model(image)
    boxes = []
    for result in results:
        for r in result.boxes:
            x1, y1, x2, y2 = r.xyxy[0].numpy()  # Get coordinates
            w = x2 - x1
            h = y2 - y1
            class_id = int(r.cls[0])
            confidence = float(r.conf[0])
            boxes.append([int(x1), int(y1), int(w), int(h), class_id, confidence])
    return boxes

# Function to compare objects between two images with proximity threshold
def compare_objects(base_boxes, test_boxes, threshold=20):
    appeared = []
    disappeared = []
    moved = []
    
    def boxes_are_close(box1, box2, threshold):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        return abs(x1 - x2) <= threshold and abs(y1 - y2) <= threshold and abs(w1 - w2) <= threshold and abs(h1 - h2) <= threshold

    base_set = set(tuple(box[:4]) for box in base_boxes)
    test_set = set(tuple(box[:4]) for box in test_boxes)
    
    for test_box in test_boxes:
        print('test_box',test_box)
        if not any(boxes_are_close(test_box[:4], base_box[:4], threshold) for base_box in base_boxes):
            appeared.append(test_box)
    
    for base_box in base_boxes:
        if not any(boxes_are_close(base_box[:4], test_box[:4], threshold) for test_box in test_boxes):
            disappeared.append(base_box)

    for base_box in base_boxes:
        for test_box in test_boxes:
            if boxes_are_close(base_box[:4], test_box[:4], threshold) and base_box[:2] != test_box[:2] and base_box[2:4] == test_box[2:4]:
                moved.append((base_box, test_box))
    
    return appeared, disappeared, moved

def display_side_by_side(image1, image2,appeared, disappeared):
    height1, width1, _ = image1.shape
    height2, width2, _ = image2.shape
    max_width = max(width1, width2)
    canvas = np.zeros((height1+height2, max_width*2, 3), dtype=np.uint8)
    canvas[:height1, :max_width, :] = image1
    canvas[height1:height1 + height2, :max_width, :] = image2
    asd=log_changes(appeared, disappeared,height1 + height2,max_width)
    canvas[:height1 + height2, max_width:max_width*2, :] = asd
    cv2.imshow('Side by Side', canvas)

def log_changes(appeared, disappeared,h,w):
    log_image = np.ones((h,w, 3), dtype=np.uint8) * 255
    k = 20
    n=0
    for i, box in enumerate(appeared):
        x, y, w, h, class_id, confidence = box
        log_text = f"Appeared {i+1}: {model.names[class_id]}: {confidence:.2f}"
        cv2.putText(log_image, log_text, (10, k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        k += 20
        n=n+k
    n += 20  # Add some space between appeared and disappeared logs
    for i, box in enumerate(disappeared):
        x, y, w, h, class_id, confidence = box
        log_text = f"Disappeared {i+1}: {model.names[class_id]}: {confidence:.2f}"
        cv2.putText(log_image, log_text, (10, k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        k += 20
    return log_image

def change_index(current_index, step, list_length):
    new_index = (current_index + step) % list_length
    return new_index

# Process each frame
for i in range(200,test_images.shape[0]):
    canvas[:, :, :] = 0
    k = change_index(i, 1200, 2735)
    #print('k:', k, 'i:', i)
    base_image = base_images[k]
    test_image = test_images[i]
    
    if base_image.shape[2] == 4:
        base_image = cv2.cvtColor(base_image, cv2.COLOR_BGRA2BGR)
    if test_image.shape[2] == 4:
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGRA2BGR)
    
    base_boxes = detect_objects_yolo(base_image)
    test_boxes = detect_objects_yolo(test_image)
    
    appeared, disappeared, moved = compare_objects(base_boxes, test_boxes)
    
    for box in test_boxes:
        x, y, w, h, class_id, confidence = box
        label = f"{model.names[class_id]}: {confidence:.2f}"
        cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(test_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    for box in base_boxes:
        x, y, w, h, class_id, confidence = box
        label = f"{model.names[class_id]}: {confidence:.2f}"
        cv2.rectangle(base_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(base_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    for box in appeared:
        x, y, w, h = box[:4]
        cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #cv2.putText(test_image, f'Appeared:{box}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    for box in disappeared:
        x, y, w, h = box[:4]
        cv2.rectangle(base_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #cv2.putText(base_image, f'Disappeared:{box}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    for base_box, test_box in moved:
        x, y, w, h = test_box[:4]
        cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        #cv2.putText(test_image, 'Moved', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    display_side_by_side(base_image, test_image,appeared, disappeared)
    #log_changes(appeared, disappeared)
    
    #x_in_map = int(test_gps[i][0] * 150) + canvas.shape[1] // 2
    #y_in_map = canvas.shape[0] // 2 - int(test_gps[i][1] * 150) - canvas.shape[0] // 4
    #cv2.circle(canvas, (x_in_map, y_in_map), 12, (0, 0, 255), 2)
    #angle = np.arctan2(test_compass[i][1], test_compass[i][0]) - np.pi / 2
    #nx_in_map = x_in_map + int(18 * np.cos(angle))
    #ny_in_map = y_in_map + int(18 * np.sin(angle))
    #cv2.line(canvas, (x_in_map, y_in_map), (nx_in_map, ny_in_map), (0, 255, 0), 1)
    #cv2.imshow('Map', canvas)
    
    k = cv2.waitKey(10)
    if k == 113 or k == 27:
        break

cv2.destroyAllWindows()


# In[ ]:




