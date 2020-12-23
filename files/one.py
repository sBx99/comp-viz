import cv2

# IMAGES
def image():
    img = cv2.imread('./assets/image.png')
    while True:
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# VIDEOS
def video():
    cap = cv2.VideoCapture('./assets/video.mp4')

    while True:
        success, img = cap.read()
        cv2.imshow('video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# WEBCAM
def webcam():
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    # cam.set(10, 100)

    while True:
        success, img = cam.read()
        cv2.imshow('video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
