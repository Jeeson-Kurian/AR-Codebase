# pip install mediapipe opencv-python pillow numpy

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageFilter
import time

# Load All 11 Hairstyles
hairstyles = [f"hairstyles/hairstyle{i}.png" for i in range(1, 12)]
current_style = 0

# MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.7)

# Webcam
cap = cv2.VideoCapture(0)
prev_time = time.time()

def calculate_iou(box1, box2):
    """Compute Intersection over Union between two boxes"""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    unionArea = box1Area + box2Area - interArea
    if unionArea == 0:
        return 0
    return interArea / unionArea

def get_hair_box(center_top, forehead_width, hair_img, adjustment_x, adjustment_y):
    hair_width = int(forehead_width * 2.0)
    hair_height = int(hair_width * hair_img.height / hair_img.width * 0.8)
    x_offset = max(0, center_top[0] - hair_width // 2 + adjustment_x)
    y_offset = max(0, center_top[1] - hair_height // 2 + adjustment_y)
    return [x_offset, y_offset, x_offset + hair_width, y_offset + hair_height]

print("Press keys 1-9, 0 for hairstyle10, '-' for hairstyle11. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    key = cv2.waitKey(1) & 0xFF
    if key in [ord(str(k)) for k in range(1, 10)]:
        current_style = key - ord('1')
    elif key == ord('0'):
        current_style = 9
    elif key == ord('-'):
        current_style = 10
    elif key == ord('q'):
        break

    iou_text = "IoU: N/A"
    landmark_text = "Landmarks: 0"

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        coords = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in face_landmarks.landmark]

        landmark_text = f"Landmarks: {len(coords)}"

        center_top = coords[10]
        left_temple = coords[127]
        right_temple = coords[356]
        forehead_box = [left_temple[0], left_temple[1], right_temple[0], right_temple[1] + 50]

        # Evaluate best hairstyle based on IoU
        best_iou = 0
        best_style = current_style
        adjustment_x = -20
        adjustment_y = -100
        for idx, path in enumerate(hairstyles):
            hair_png = Image.open(path).convert("RGBA")
            hair_box = get_hair_box(center_top, np.linalg.norm(np.array(right_temple) - np.array(left_temple)), hair_png, adjustment_x, adjustment_y)
            iou = calculate_iou(forehead_box, hair_box)
            if iou > best_iou:
                best_iou = iou
                best_style = idx

        # If current IoU is too low, auto-recommend better style
        if best_iou > 0.35 and best_style != current_style:
            current_style = best_style

        iou_text = f"IoU: {best_iou:.2f}"
        hair_png = Image.open(hairstyles[current_style]).convert("RGBA")

        # Calculate angle
        dx = right_temple[0] - left_temple[0]
        dy = right_temple[1] - left_temple[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # Resize and rotate
        forehead_width = int(np.linalg.norm(np.array(right_temple) - np.array(left_temple)))
        hair_width = int(forehead_width * 2.0)
        hair_height = int(hair_width * hair_png.height / hair_png.width * 0.8)
        resized_hair = hair_png.resize((hair_width, hair_height)).rotate(-angle, expand=True)

        # Blend
        blurred_alpha = resized_hair.split()[3].filter(ImageFilter.GaussianBlur(radius=1))
        soft_hair = resized_hair.copy()
        soft_hair.putalpha(blurred_alpha)

        x_offset = max(0, center_top[0] - hair_width // 2 + adjustment_x)
        y_offset = max(0, center_top[1] - hair_height // 2 + adjustment_y)

        base_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        overlay_layer = Image.new("RGBA", base_img.size)
        overlay_layer.paste(soft_hair, (x_offset, y_offset), mask=soft_hair)
        final = Image.alpha_composite(base_img, overlay_layer)
        frame = cv2.cvtColor(np.array(final.convert("RGB")), cv2.COLOR_RGB2BGR)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display metrics
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Style: {current_style + 1}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, iou_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(frame, landmark_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    cv2.imshow('🎯 Real-Time AR Hairstyle Overlay with Metrics + Auto-Adjust', frame)

cap.release()
cv2.destroyAllWindows()
