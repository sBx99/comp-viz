#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2

img = cv2.imread('example.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# image, threshold value, max value, type of thresholding
_, thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)

cv2.imshow('img', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
