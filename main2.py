import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer
import time

# Initialize alert sound
mixer.init()
sound = mixer.Sound('alarm2.wav')

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Load trained model
model = load_model(os.path.join(r'C:\Users\aayue\OneDrive\Desktop\Projects\Drowsiness Detection\models\model.h5'))

# Labels for prediction
lbl = ['Close', 'Open']

# Set video capture with reduced resolution for better performance
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Font settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_color_open = (0, 255, 0)
font_color_closed = (0, 0, 255)
font_thickness = 2

# Variables to track drowsiness
score = 0
alert_display = False
alert_start_time = None
alert_duration = 2  # Display alert for 2 seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces and eyes
    faces = face_cascade.detectMultiScale(gray, minNeighbors=3, scaleFactor=1.1, minSize=(25, 25))
    eyes = eye_cascade.detectMultiScale(gray, minNeighbors=1, scaleFactor=1.1)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    for (x, y, w, h) in eyes:
        eye = frame[y:y + h, x:x + w]
        eye = cv2.resize(eye, (80, 80))
        eye = eye / 255.0
        eye = eye.reshape(80, 80, 3)
        eye = np.expand_dims(eye, axis=0)
        prediction = model.predict(eye)
        
        if prediction[0][0] > 0.30:  # Eye Closed
            score += 1
            cv2.putText(frame, "Closed", (20, 50), font, font_scale, font_color_closed, font_thickness, cv2.LINE_AA)
        
            if score > 7 and not alert_display:
                sound.play()
                alert_display = True
                alert_start_time = time.time()
        
        elif prediction[0][1] > 0.70:  # Eye Open
            score = max(0, score - 1)
            cv2.putText(frame, "Open", (20, 50), font, font_scale, font_color_open, font_thickness, cv2.LINE_AA)
        
    # Alert Message
    if alert_display:
        elapsed_time = time.time() - alert_start_time
        if elapsed_time < alert_duration:
            cv2.putText(frame, "ALERT! EYES CLOSED!", (150, 100), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            alert_display = False  # Reset alert display after duration
    
    # Display instructions
    cv2.putText(frame, "Press 'E' to exit", (450, 470), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Show frame
    cv2.imshow('Drowsiness Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()