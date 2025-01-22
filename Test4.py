import cv2
import numpy as np

cap = cv2.VideoCapture(0)
green = [0, 128, 0]


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

    #Lloyds Algorithm
    pixels = frame.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 8  
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(frame.shape)

    hsv_frame = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([10, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([70, 255, 255], dtype=np.uint8)
    lower_green = np.array([40, 40, 40], dtype=np.uint8)
    upper_green = np.array([80, 255, 255], dtype=np.uint8)

    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

    value_channel = hsv_frame[:, :, 2]
    blurred_value = cv2.GaussianBlur(value_channel, (5, 5), 0)
    adaptive_threshold = cv2.adaptiveThreshold(
        blurred_value, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    yellow_blurred = cv2.GaussianBlur(yellow_mask, (5, 5), 0)
    yellow_edges = cv2.Canny(yellow_blurred, 50, 150)
    green_blurred = cv2.GaussianBlur(green_mask, (5, 5), 0)
    green_edges = cv2.Canny(green_blurred, 50, 150)

    yellow_result = cv2.bitwise_and(adaptive_threshold, yellow_mask)
    yellow_contours = cv2.findContours(yellow_edges, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    green_result = cv2.bitwise_and(adaptive_threshold, green_mask)
    green_contours, _ = cv2.findContours(green_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    yellow_hexagons = []
    green_hexagons = []

    for yellow_contour in yellow_contours:
        epsilon = 0.02 * cv2.arcLength(yellow_contour, True)
        approx = cv2.approxPolyDP(yellow_contour, epsilon, True)
        if len(approx) == 6:
            yellow_hexagons.append(approx)
    for green_contour in green_contours:
        epsilon = 0.02 * cv2.arcLength(green_contour, True)
        approx = cv2.approxPolyDP(green_contour, epsilon, True)
        if len(approx) == 6:
            green_hexagons.append(approx)
    thing_result = frame.copy
    cv2.drawContours(frame, yellow_hexagons, -1, (0, 255, 255), 2)
    cv2.drawContours(frame, green_hexagons, -1, (0, 255, 0), 2)


    cv2.imshow('frame', frame)
    cv2.imshow('result', segmented_image)
    cv2.imshow('Segmented Image', yellow_result)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
