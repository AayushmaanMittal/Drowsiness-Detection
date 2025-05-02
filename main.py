import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer
import time

mixer.init()
sound = mixer.Sound('alarm2.wav')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
model = load_model(os.path.join(r'C:\Users\aayue\OneDrive\Desktop\Projects\Drowsiness Detection\models\model.h5'))

lbl = ['Close', 'Open']

path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
alert_timer = 0  # Counter to manage alert display duration

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, minNeighbors=3, scaleFactor=1.1, minSize=(25, 25))
    eyes = eye_cascade.detectMultiScale(gray, minNeighbors=1, scaleFactor=1.1)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    # Add title and instructions
    
    cv2.putText(frame, "Press 'E' to exit", (width - 200, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    for (x, y, w, h) in eyes:
        eye = frame[y:y + h, x:x + w]
        eye = cv2.resize(eye, (80, 80))
        eye = eye / 255
        eye = eye.reshape(80, 80, 3)
        eye = np.expand_dims(eye, axis=0)
        prediction = model.predict(eye)

        # Condition for Close
        if prediction[0][0] > 0.30:
            cv2.putText(frame, "Closed", (10, height - 20), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
            score = score + 1

            if score > 10:
                try:
                    sound.play()
                    cv2.putText(frame, "Alert: Eyes Closed!", (10, 60), font, 2.5, (0, 0, 255), 2, cv2.LINE_AA)
                    # Add code to display alert text
                    alert_timer = 1  # Set the timer to start displaying the alert
                except:
                    pass

        # Condition for Open
        elif prediction[0][1] > 0.70:
            score = score - 1
            if score < 0:
                score = 0
            cv2.putText(frame, "Open", (10, height - 20), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()
