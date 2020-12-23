#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2

# load the pre-trained cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# load image
img = cv2.imread('obama.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# run face detection
# detectMultiScale(img, scale factor, number of neighbors)
face_coords = face_cascade.detectMultiScale(img_gray, 1.1, 8)

# show faces
i = 0
for face_coord in face_coords:
    x, y, w, h = face_coord
    # draw the rectangle on the main image
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
    
    # extract faces from main image
    # OpenCV and numpy: y <-> row and x <-> col
    face = img[y:y+h, x:x+w]
    
    # show face0 and face1, etc.
    cv2.imshow('face{}'.format(i), face)
    i += 1
    
# shows the image and waits for a keypress
cv2.imshow('faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
