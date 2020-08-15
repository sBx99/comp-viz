#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import sys

# load the pre-trained cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# load image
img = cv2.imread('obama.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# run face detection
faces = face_cascade.detectMultiScale(img_gray, 1.1, 8)
# check number of faces
if len(faces) != 2:
    sys.exit('Please input an image with EXACTLY 2 faces!')

# retrieve dimensions of faces
x1, y1, w1, h1 = faces[0]
x2, y2, w2, h2 = faces[1]

# extract both faces from the image
face1 = img[y1:y1+h1, x1:x1+w1]
face2 = img[y2:y2+h2, x2:x2+w2]

# resize face 2 into face 1's dimensions and vice-versa
face2 = cv2.resize(face2, (w1, h1))
face1 = cv2.resize(face1, (w2, h2))

# replace face 2 and with face 1 in the image
img[y2:y2+h2, x2:x2+w2] = face1

# replace face 1 and with face 2 in the image
img[y1:y1+h1, x1:x1+w1] = face2

# show face swapping
cv2.imshow('swap', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
