# Install dependencies if not already done
# pip install mediapipe opencv-python pillow numpy
#demo

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageFilter

# Load Multiple Hairstyles
hairstyles = [
    "hairstyles/hairstyle1.png",
    "hairstyles/hairstyle2.png",
    "hairstyles/hairstyle3.png",
    "hairstyles/hairstyle4.png"
]
current_style = 0  # Default style index

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.7)

# Open Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Handle Key Press for Changing Styles Dynamically
    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        current_style = 0
    elif key == ord('2'):
        current_style = 1
    elif key == ord('3'):
        current_style = 2
    elif key == ord('4'):
        current_style = 3
    elif key == ord('q'):
        break  # Quit program

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        coords = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in face_landmarks.landmark]

        # Forehead Landmarks
        center_top = coords[10]
        left_temple = coords[127]
        right_temple = coords[356]

        # Load Selected Hairstyle
        hair_png = Image.open(hairstyles[current_style]).convert("RGBA")

        # Calculate Size & Rotation
        forehead_width = int(np.linalg.norm(np.array(right_temple) - np.array(left_temple)))
        hair_width = int(forehead_width * 2.0)
        hair_height = int(hair_width * hair_png.height / hair_png.width * 0.8)

        dx = right_temple[0] - left_temple[0]
        dy = right_temple[1] - left_temple[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # Resize and Rotate Hair
        resized_hair = hair_png.resize((hair_width, hair_height)).rotate(-angle, expand=True)
        blurred_alpha = resized_hair.split()[3].filter(ImageFilter.GaussianBlur(radius=1))
        soft_hair = resized_hair.copy()
        soft_hair.putalpha(blurred_alpha)

        # Placement Calculation
        adjustment_x = -20
        adjustment_y = -100
        x_offset = center_top[0] - hair_width // 2 + adjustment_x
        y_offset = center_top[1] - hair_height // 2 + adjustment_y

        # Boundary Checks
        x_offset = max(0, x_offset)
        y_offset = max(0, y_offset)

        # Overlay
        base_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        overlay_layer = Image.new("RGBA", base_img.size)
        overlay_layer.paste(soft_hair, (x_offset, y_offset), mask=soft_hair)

        final = Image.alpha_composite(base_img, overlay_layer)
        frame = cv2.cvtColor(np.array(final.convert("RGB")), cv2.COLOR_RGB2BGR)

    # Display Output
    cv2.imshow('🎯 Real-Time AR Hairstyle Overlay', frame)

cap.release()
cv2.destroyAllWindows()
