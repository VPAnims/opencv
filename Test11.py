import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

while True:
    ret, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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
    else:
        color = "white"
    cv2.putText(hsv_frame, color, (10, 50), 0, 1, (0, 0, 0), 2)
    cv2.circle(hsv_frame, (cx, cy), 5, (255, 255, 255), 1)
    cv2.imshow('frame', hsv_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

