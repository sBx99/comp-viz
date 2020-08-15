#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from dataset import load_data, reorder

def preprocess(img):
    """Preprocess the input image for image similarity
    
    Args:
        img (RGB image): input image to be processed
    
    Returns:
        binary image: processed image to pass through the model
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 4)
    return thresh

def similarity(A, B):
    """Computes a similarity metric for two cards
    
    Args:
        A (binary image): first card
        B (binary image): second card
    
    Returns:
        int: similarity metric (smaller is more similar)
    """
    diff = cv2.absdiff(A, B)
    
    _, diff = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY)
    
    return np.sum(diff)

def extract_cards(img, num_cards):
    """Extracts num_cards cards from img
    
    Args:
        img (RGB image): image to extract cards from
        num_cards (int): Number of cards to extract
    
    Returns:
        list: list of images of cards
    """
    cards = []
    
    # preprocess image for contour detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
    
    # find contours and get num_cards number of largest contours
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:num_cards]
    
    # new dimensions after affine transform
    new_dim = np.array([[0,0], [449, 0], [449, 449], [0,449]], np.float32)
    
    for card in contours:
        # approximate the contour with a rectangle
        epsilon = 0.01 * cv2.arcLength(card, True)
        approx = cv2.approxPolyDP(card, epsilon, True)
        approx = reorder(approx)
        
        # apply the affine transform
        transform = cv2.getPerspectiveTransform(approx, new_dim)
        warp = cv2.warpPerspective(img, transform, (450, 450))
        cards.append(warp)
    
    return cards

class Model(object):
    def fit(self, X, y):
        """Trains a nearest-neighbor classifier
        
        Args:
            X (list of images): training data, i.e., list of card images
            y (list of tuples): training labels, i.e., list of (rank,suit) tuples
        """
        self.X = X
        self.y = y
    
    def predict(self, X):
        """Predicts the rank and suit of an input card
        
        Args:
            X (list of images or image): Image(s) to process
        
        Returns:
            list of tuples: list of predictions (rank, suit)
        """
        if type(X) is not list:
            X = [X]
        
        predictions = []
        for x in X:
            # keep track of closest match
            prediction = None
            closest_match = float('inf')
            
            # iterate through our training images
            for i, training_img in enumerate(self.X):
                card_similarity = similarity(training_img, x)
                if card_similarity <= closest_match:
                    # update closest match
                    prediction = self.y[i]
                    closest_match = card_similarity
            
            predictions.append(prediction)
        
        return predictions

if __name__ == '__main__':
    # load input data
    filename = 'test.jpg'
    num_cards = 4
    
    test_img = cv2.imread(filename)
    
    # load training data
    X, y = load_data()
    
    # preprocess training inputs
    X = list(map(preprocess, X))
    """
    for i, x in enumerate(X):
        cv2.imshow(str(y[i]), x)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    """
    
    # create our model and train it
    model = Model()
    model.fit(X, y)
    
    # extract cards from test image and evaluate model
    cards = extract_cards(test_img, num_cards=num_cards)
    for i, card in enumerate(cards):
        # preprocess test card
        processed_card = preprocess(card)
        
        # use model to prediction
        prediction = model.predict(processed_card)[0]
        
        cv2.imshow('{}: {}'.format(i, prediction), card)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
