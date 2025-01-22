import cv2
import numpy as np

cap = cv2.VideoCapture(0)

green = [0,128,0]
purple = [193,126,127]
yellow = [65,176,226]


def get_color_limits(color):
    BGR = np.uint8([[color]])
    HSV = cv2.cvtColor(BGR, cv2.COLOR_BGR2HSV)

    lower_limit = HSV[0][0][0] - 20, 100, 100
    upper_limit = HSV[0][0][0] + 20, 255, 255

    lower_limit = np.array(lower_limit, dtype=np.uint8)
    upper_limit = np.array(upper_limit, dtype=np.uint8)
    return lower_limit, upper_limit

while True:

    ret, frame = cap.read()

    #lower_green, upper_green = get_color_limits(color=purple )
    #lower_purple, upper_purple = get_color_limits(color=purple)
    lower_purple = np.array([83, 199, 240], dtype=np.uint8)    
    upper_purple = np.array([142, 212, 250], dtype=np.uint8)
    pixels = frame.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 7  
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(frame.shape)

    hsv = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, lower_purple, upper_purple)

    value_channel = hsv[:, :, 2]
    blurred_value = cv2.GaussianBlur(value_channel, (5, 5), 0)
    adaptive_threshold = cv2.adaptiveThreshold(
        blurred_value, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    green_result = cv2.bitwise_and(adaptive_threshold, green_mask)

    cv2.imshow('Adaptive Thresholding', adaptive_threshold) 
    cv2.imshow('Original Image', segmented_image)
    cv2.imshow('e', green_result)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()