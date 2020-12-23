import numpy as np
import cv2

def reorder(corners):
    """Reorder points in a rectangle in clockwise order to be consistent with OpenCV
    
    Args:
        corners (ndarray): corners of rectangle. Must be a 4x2 numpy matrix
    
    Returns:
        ndarray: A re-oriented rectangle in accordance with OpenCV
    """
    corners = corners.reshape((4,2))
    oriented_corners = np.zeros((4,2), np.float32)

    add = corners.sum(1)
    oriented_corners[0] = corners[np.argmin(add)]
    oriented_corners[2] = corners[np.argmax(add)]

    diff = np.diff(corners, 1)
    oriented_corners[1] = corners[np.argmin(diff)]
    oriented_corners[3] = corners[np.argmax(diff)]

    return oriented_corners

def extract_training_cards(img, num_cards):
    """Extracts num_cards cards from img
    
    Args:
        img (RGB image): image to extract cards from
        num_cards (int): Number of cards to extract
    
    Returns:
        list: list of images of cards
    """
    cards = []

    # preprocess image for contour extraction
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)

    # find contours and get the num_cards number of largest contours
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:num_cards]

    # new dimensions after the affine transform
    new_dim = np.array([[0,0], [449, 0], [449, 449], [0, 449]], np.float32)

    for card in contours:
        # approximate a rectangle
        epsilon = 0.01*cv2.arcLength(card, True)
        approx = cv2.approxPolyDP(card, epsilon, True)
        approx = reorder(approx)

        # apply the affine transform
        transform = cv2.getPerspectiveTransform(approx, new_dim)
        warp = cv2.warpPerspective(img, transform, (450, 450))
        cards.append(warp)

    return cards

def load_data(training_labels_filename='train.csv', training_image_filename='train.png', num_cards=56):
    """Loads training data
        training_labels_filename (str, optional): Defaults to 'train.tsv'. Ground-truth training labels
        training_image_filename (str, optional): Defaults to 'train.png'. Ground-truth training image (containing all training examples)
        num_cards (int, optional): Defaults to 56. Number of cards in the training image
    
    Returns:
        (X, y): a list of RGB images and a list of labels
    """
    X = []
    y = []
  
    # open training labels file
    labels = {}
    with open(training_labels_filename) as f:
        for line in f:
            # load training data into a dictionary
            key, rank, suit = line.strip().split(',')
            labels[int(key)] = (rank, suit)
    
    # load training image and extract all of the training cards
    training_img = cv2.imread(training_image_filename)
    for i, card in enumerate(extract_training_cards(training_img, num_cards)):
        X.append(card)
        y.append(labels[i])
        """
        cv2.imshow(str(labels[i]), card)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
  
    return X, y