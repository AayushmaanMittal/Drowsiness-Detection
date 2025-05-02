import cv2
import os
import time
import numpy as np
from pygame import mixer
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

# Initialize alert sound
mixer.init()
sound = mixer.Sound('alarm2.wav')

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Load trained eye state model
model = load_model(os.path.join(r'C:\Users\aayue\OneDrive\Desktop\Projects\Drowsiness Detection\models\model.h5'))

# MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Mouth landmark indices for MAR
MOUTH_LANDMARKS = [13, 14, 78, 308]  # Upper, lower, left, right lip points

# Function to calculate Mouth Aspect Ratio (MAR)
def calculate_mar(landmarks, frame_width, frame_height):
    upper = landmarks[13]
    lower = landmarks[14]
    left = landmarks[78]
    right = landmarks[308]
    
    # Convert to pixel coordinates
    upper = np.array([int(upper.x * frame_width), int(upper.y * frame_height)])
    lower = np.array([int(lower.x * frame_width), int(lower.y * frame_height)])
    left = np.array([int(left.x * frame_width), int(left.y * frame_height)])
    right = np.array([int(right.x * frame_width), int(right.y * frame_height)])

    # Vertical & horizontal distances
    vertical = np.linalg.norm(upper - lower)
    horizontal = np.linalg.norm(left - right)
    
    mar = vertical / horizontal
    return mar

# Labels
lbl = ['Close', 'Open']

# Video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

font = cv2.FONT_HERSHEY_SIMPLEX

# Tracking variables
score = 0
alert_display = False
alert_start_time = None
alert_duration = 2

yawn_alert = False
yawn_alert_time = None
yawn_threshold = 0.6  # Adjust if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face & eye detection
    faces = face_cascade.detectMultiScale(gray, minNeighbors=3, scaleFactor=1.1, minSize=(25, 25))
    eyes = eye_cascade.detectMultiScale(gray, minNeighbors=1, scaleFactor=1.1)

    # Yawn detection using MediaPipe
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        mar = calculate_mar(landmarks, w, h)

        if mar > yawn_threshold:
            yawn_alert = True
            yawn_alert_time = time.time()

    # Draw face rectangles
    for (x, y, w_face, h_face) in faces:
        cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), (255, 0, 0), 2)

    # Eye state prediction
    for (x, y, w_eye, h_eye) in eyes:
        eye = frame[y:y + h_eye, x:x + w_eye]
        eye = cv2.resize(eye, (80, 80))
        eye = eye / 255.0
        eye = eye.reshape(80, 80, 3)
        eye = np.expand_dims(eye, axis=0)
        prediction = model.predict(eye)

        if prediction[0][0] > 0.30:
            score += 1
            cv2.putText(frame, "Closed", (20, 50), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            if score > 7 and not alert_display:
                sound.play()
                alert_display = True
                alert_start_time = time.time()
        elif prediction[0][1] > 0.70:
            score = max(0, score - 1)
            cv2.putText(frame, "Open", (20, 50), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Eye alert message
    if alert_display:
        if time.time() - alert_start_time < alert_duration:
            cv2.putText(frame, "ALERT! EYES CLOSED!", (150, 100), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            alert_display = False

    # Yawn alert message
    if yawn_alert and (time.time() - yawn_alert_time < 2):
        cv2.putText(frame, "YAWN DETECTED!", (150, 150), font, 1, (0, 165, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, "Please take rest / wash your face then drive", (50, 190), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    else:
        yawn_alert = False

    cv2.putText(frame, "Press 'E' to exit", (450, 470), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('Drowsiness + Yawn Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()
