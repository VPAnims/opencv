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

    pixels = frame.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 4
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(frame.shape)

    hsv = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2HSV)
    lower_green, upper_green = get_color_limits(color=green)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    block_size = 11
    constant = 2
    adaptive_thresholdMask= cv2.adaptiveThreshold(green_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant)
    result = cv2.bitwise_and(segmented_image, frame, mask = adaptive_thresholdMask)


    #cv2.imshow('frame', frame)
    cv2.imshow('thresholding', result)
    cv2.imshow('gmae', segmented_image)
    cv2.imshow("skdjf", adaptive_thresholdMask)
    cv2.imshow('hd', green_mask)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()