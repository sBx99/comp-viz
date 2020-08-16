#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np

from dataset import load_data

# create face detector and load data
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
(X_train, y_train), (X_test, y_test) = load_data(face_cascade)

# create and train face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(X_train, np.array(y_train))

# open webcam
video_cap = cv2.VideoCapture(0)
while True:
    # get a frame from webcam
    _, frame = video_cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # run face detection
    faces = face_cascade.detectMultiScale(gray)
    for face in faces:
        # extract face from frame
        x, y, w, h = face
        face_region = gray[y:y+h, x:x+w]
        
        # run face recognition
        predicted_person, _ = face_recognizer.predict(face_region)
        
        # draw rectangle over video feed
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, str(predicted_person), (x,y), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255))
    
    # show annotated frame
    cv2.imshow('face recognition', frame)
    
    # quit app
    if cv2.waitKey(1) == ord('q'):
        break
# end while

# close webcam and release resources
video_cap.release()
cv2.destroyAllWindows()
