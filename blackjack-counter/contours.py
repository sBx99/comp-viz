#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2

img = cv2.imread('wallet.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7,7), 0)

# image, threshold value, max value, type of thresholding
_, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

_, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contour = contours[32]

epsilon = 0.01 * cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, epsilon, True)

cv2.drawContours(img, [approx], -1, (0,255,0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
