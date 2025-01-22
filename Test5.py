import cv2
import numpy as np

cap = cv2.VideoCapture(0)

green = [0, 255, 0]
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

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 40, 40], dtype=np.uint8)
    upper_green = np.array([80, 255, 255], dtype=np.uint8)
    lower_green, upper_green = get_color_limits(color=green)
    lower_purple = np.array([75, 0, 130], dtype=np.uint8)
    upper_purple = np.array([221, 160, 221], dtype=np.uint8) 
    lower_yellow = np.array([10, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([70, 255, 255], dtype=np.uint8)

    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    purple_mask = cv2.inRange(hsv_frame, lower_purple, upper_purple)
    yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
    value_channel = hsv_frame[:, :, 2]
    blurred_value = cv2.GaussianBlur(value_channel, (5, 5), 0)
    adaptive_threshold = cv2.adaptiveThreshold(
        blurred_value, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )


    green_result = cv2.bitwise_and(adaptive_threshold, green_mask)
    purple_result = cv2.bitwise_and(adaptive_threshold, purple_mask)
    yellow_result = cv2.bitwise_and(adaptive_threshold, yellow_mask)
    green_blurred = cv2.GaussianBlur(green_mask, (5, 5), 0)
    green_edges = cv2.Canny(green_blurred, 50, 150)
    purple_blurred = cv2.GaussianBlur(purple_mask, (5, 5), 0)
    purple_edges = cv2.Canny(purple_blurred, 50, 150)
    yellow_blurred = cv2.GaussianBlur(yellow_mask, (5, 5), 0)
    yellow_edges = cv2.Canny(yellow_blurred, 50, 150)

    green_contours, _ = cv2.findContours(green_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    purple_contours, _ = cv2.findContours(purple_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    yellow_contours, _ = cv2.findContours(yellow_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    green_hexagons = []
    yellow_hexagons = []
    purple_hexagons = []
    for contour in green_contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 6:
            green_hexagons.append(approx)
    for contour in purple_contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 6:
            purple_hexagons.append(approx)
    for contour in yellow_contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 6:
            yellow_hexagons.append(approx)
    thing_result = frame.copy()
    cv2.drawContours(thing_result, yellow_hexagons, -1, (0, 255, 255), 2)

    cv2.drawContours(thing_result, purple_hexagons, -1, (128, 0, 128), 2)

    cv2.imshow('Original Frame', frame)
    cv2.imshow('Green Color Mask', purple_mask)
    cv2.imshow('Adaptive Thresholding', adaptive_threshold)
    cv2.imshow('Final Result', purple_result)
    cv2.imshow('kj', thing_result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
#4097 5813 3212