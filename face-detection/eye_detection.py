#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2

# load the pre-trained cascade classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# load image
img = cv2.imread('obama.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# run face detection
# detectMultiScale(img, scale factor, number of neighbors)
face_coords = face_cascade.detectMultiScale(img_gray, 1.1, 8)

# show faces
for face_coord in face_coords:
    x, y, w, h = face_coord
    # draw the rectangle on the main image
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

# run eye detection (scale factor = 1.1, minNeighbors=3)
eye_coords = eye_cascade.detectMultiScale(img_gray, 1.1, 6)

# show eyes
for eye_coord in eye_coords:
    x, y, w, h = eye_coord
    # draw the rectangle on the main image
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
    
# shows the image and waits for a keypress
cv2.imshow('faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
