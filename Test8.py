import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0)

yellow = [84,171,220]

# green = rgba(240,199,83,255), rgba(250,212,142,255), rgba(246,210,131,255), rgba(246,210,131,255)
def get_color_limits(color):
    BGR = np.uint8([[color]])
    HSV = cv2.cvtColor(BGR, cv2.COLOR_BGR2HSV)

    lower_limit = HSV[0][0][0] - 40, 100, 100
    upper_limit = HSV[0][0][0] + 40, 255, 255

    lower_limit = np.array(lower_limit, dtype=np.uint8)
    upper_limit = np.array(upper_limit, dtype=np.uint8)
    return lower_limit, upper_limit

while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    lower_yellow, upper_yellow  = get_color_limits(color=yellow)

    pixels = frame.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 6, 0.2)
    k = 4
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 7, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(frame.shape)

    hsv = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    value_channel = hsv[:, :, 2]
    blurred_value = cv2.GaussianBlur(value_channel, (5, 5), 0)
    adaptive_threshold = cv2.adaptiveThreshold(
        blurred_value, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    yellow_result = cv2.bitwise_and(adaptive_threshold, yellow_mask)

    cv2.imshow('Adaptive Thresholding', adaptive_threshold) 
    cv2.imshow('frame', segmented_image)
    cv2.imshow('hsv', hsv)
    cv2.imshow('e', yellow_result)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
