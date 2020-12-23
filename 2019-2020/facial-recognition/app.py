#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import argparse
import time
import numpy as np

from dataset import load_data

# define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--classifier', '-c', choices=['lbp', 'eigen', 'fisher'], default='lbp')
args = parser.parse_args()

print(args.classifier)

# create face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# load all data
(X_train, y_train), (X_test, y_test) = load_data(face_cascade)

if args.classifier in ['eigen', 'fisher']:
    # resize images to one size
    X_train = [cv2.resize(img, (128, 128)) for img in X_train]
    X_test = [cv2.resize(img, (128, 128)) for img in X_test]

# create face recognizer
if args.classifier == 'lbp':
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
elif args.classifier == 'eigen':
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
elif args.classifier == 'fisher':
    face_recognizer = cv2.face.FisherFaceRecognizer_create()

# train face recognizer on the training set
start = time.time()
face_recognizer.train(X_train, np.array(y_train))
end = time.time()
print('Training time: {}s'.format(end - start))

# evaluate face recognizer on the test set
prediction_time = 0.
accuracy = 0.
for i, test_img in enumerate(X_test):
    start = time.time()
    y_pred, confidence = face_recognizer.predict(test_img)
    end = time.time()
    prediction_time += (end - start)
    if y_pred == y_test[i]:
        accuracy += 1

accuracy = accuracy / len(X_test)
print('Accuracy: {:.4f}'.format(accuracy))
prediction_time = prediction_time  / len(X_test)
print('Average prediction time: {}s'.format(prediction_time))
