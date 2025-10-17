# Install required packages if needed
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

# Start Webcam
cap = cv2.VideoCapture(0)
prev_time = time.time()

print("Press keys 1-9, 0 for hairstyle10, '-' for hairstyle11. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Handle Key Input
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
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        coords = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in face_landmarks.landmark]

        # Key Landmarks
        center_top = coords[10]
        left_temple = coords[127]
        right_temple = coords[356]

        # Load Hairstyle
        hair_png = Image.open(hairstyles[current_style]).convert("RGBA")

        # Calculate hair size and position
        forehead_width = int(np.linalg.norm(np.array(right_temple) - np.array(left_temple)))
        hair_width = int(forehead_width * 2.0)
        hair_height = int(hair_width * hair_png.height / hair_png.width * 0.8)

        # Angle and alignment
        dx = right_temple[0] - left_temple[0]
        dy = right_temple[1] - left_temple[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # Resize and rotate hair
        resized_hair = hair_png.resize((hair_width, hair_height)).rotate(-angle, expand=True)
        blurred_alpha = resized_hair.split()[3].filter(ImageFilter.GaussianBlur(radius=1))
        soft_hair = resized_hair.copy()
        soft_hair.putalpha(blurred_alpha)

        # Position
        adjustment_x = -20
        adjustment_y = -100
        x_offset = max(0, center_top[0] - hair_width // 2 + adjustment_x)
        y_offset = max(0, center_top[1] - hair_height // 2 + adjustment_y)

        # Prepare overlay
        base_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        overlay_layer = Image.new("RGBA", base_img.size)
        overlay_layer.paste(soft_hair, (x_offset, y_offset), mask=soft_hair)

        final = Image.alpha_composite(base_img, overlay_layer)
        frame = cv2.cvtColor(np.array(final.convert("RGB")), cv2.COLOR_RGB2BGR)

        # Calculate IoU between forehead and hairstyle regions
        forehead_box = [left_temple[0], left_temple[1], right_temple[0], right_temple[1] + 50]
        hair_box = [x_offset, y_offset, x_offset + hair_width, y_offset + hair_height]

        # Intersection
        ix1 = max(forehead_box[0], hair_box[0])
        iy1 = max(forehead_box[1], hair_box[1])
        ix2 = min(forehead_box[2], hair_box[2])
        iy2 = min(forehead_box[3], hair_box[3])
        iw = max(ix2 - ix1, 0)
        ih = max(iy2 - iy1, 0)
        intersection = iw * ih

        # Union
        forehead_area = (forehead_box[2] - forehead_box[0]) * (forehead_box[3] - forehead_box[1])
        hair_area = (hair_box[2] - hair_box[0]) * (hair_box[3] - hair_box[1])
        union = forehead_area + hair_area - intersection

        if union > 0:
            iou = intersection / union
            iou_text = f"IoU: {iou:.2f}"

    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - prev_time)
    prev_time = end_time

    # Draw Metrics
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Style: {current_style + 1}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, iou_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    # Display Window
    cv2.imshow('🎯 Real-Time AR Hairstyle Overlay with Metrics', frame)

cap.release()
cv2.destroyAllWindows()
