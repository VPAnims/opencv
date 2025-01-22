import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0)

yellow = [76, 194, 238]
green = [0, 128, 0]
purple = [218, 199, 204]
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

    lower_yellow, upper_yellow  = get_color_limits(color=green)

    pixels = frame.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 8
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
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
    inner_contours, _ = cv2.findContours(contour, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for inner_contour in inner_contours:
        inner_approx = cv2.approxPolyDP(inner_contour, 0.05 * cv2.arcLength(inner_contour, True), True)
        if len(inner_approx) == 6:     
            rect = cv2.boundingRect(inner_contour)
            x, y, w, h = rect
            roi = frame[y:y+h, x:x+w]
            mean_color = cv2.mean(roi)
            color = tuple(int(c) for c in mean_color[:3])
    yellow_result = cv2.bitwise_and(adaptive_threshold, yellow_mask)

    cv2.imshow('Adaptive Thresholding', adaptive_threshold) 
    cv2.imshow('frame', segmented_image)
    cv2.imshow('hsv', hsv)
    cv2.imshow('e', yellow_result)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
