import os
import sys
import numpy as np
from PIL import Image

def _extract_face(filepath, face_cascade):
	"""Extracts face region from filepath using face_cascade
	
	Args:
		filepath (str): path to the image file
		face_cascade (cv2.CascadeClassifier): cascade classifier for faces
	
	Returns:
		ndarray: face region
	"""

	# load image and convert to numpy array
	img = Image.open(filepath).convert('L')
	img = np.array(img, np.uint8)

	# run face detection with the default parameters
	face = face_cascade.detectMultiScale(img)

	# quit if we don't have exactly 1 face!
	if len(face) != 1:
		sys.exit('Example {} does not have exactly one face!'.format(filepath))

	# extract the face from the image
	face = face[0]
	x, y, w, h = face
	face_region = img[y:y+h,x:x+w]
	return face_region

def load_data(face_cascade, data_dir='yalefaces'):
	"""Loads training and testing data
	
	Args:
		face_cascade (cv2.CascadeClassifier): cascade classifier for faces
		data_dir (str, optional): Defaults to 'yalefaces'. directory of the data
	
	Returns:
		(X_train, y_train), (X_test, y_test): training and testing data
	"""

	X_train = []
	y_train = []

	X_test = []
	y_test = []

	# split the files into training and testing sets
	training_image_files = [f for f in os.listdir(data_dir) if not f.endswith('.wink')]
	test_image_files = [f for f in os.listdir(data_dir) if f.endswith('.wink')]

	# construct training set
	for image_file in training_image_files:
		filepath = os.path.join(data_dir, image_file)

		face_region = _extract_face(filepath, face_cascade)
		person_number = int(image_file.split('.')[0].replace('subject', ''))

		X_train.append(face_region)
		y_train.append(person_number)

	# construct test set
	for image_file in test_image_files:
		filepath = os.path.join(data_dir, image_file)

		face_region = _extract_face(filepath, face_cascade)
		person_number = int(image_file.split('.')[0].replace('subject', ''))

		X_test.append(face_region)
		y_test.append(person_number)
	
	return (X_train, y_train), (X_test, y_test)
