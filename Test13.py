import cv2
import numpy as np

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

while True:
    ret, frame = cap.read()

    pixels = frame.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)
    k = 8
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    lloyd_image = segmented_image.reshape(frame.shape)
    hsv_frame = cv2.cvtColor(lloyd_image, cv2.COLOR_BGR2HSV)
    height, width, _ = frame.shape

    cx = int(width/2)
    cy = int(height/2)
    pixelcenter = hsv_frame[cy, cx]
    print(pixelcenter)
    hue = pixelcenter[0]
    color = " "
    if hue < 22:
        color = "other"
    elif hue < 33:
        color = "yellow"
    elif hue < 78:
        color = "green"
    elif hue < 131:
        color = "other"
    elif hue < 150:
        color = "purple"
    cv2.putText(frame, color, (10, 50), 0, 1, (0, 0, 0), 2)
    cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 1)
    cv2.imshow('hsv', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
