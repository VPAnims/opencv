import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0)

green = [0,128,0]

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
    
    lower_Green, upper_Green = get_color_limits(green)
    value_channel = hsv_frame[:, :, 2]
    blurred_value = cv2.GaussianBlur(value_channel, (5, 5), 0)
    adaptive_threshold = cv2.adaptiveThreshold(
        blurred_value, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, lower_Green, upper_Green)
    green_result = cv2.bitwise_and(frame, frame, mask=green_mask)

    #green_result =  cv2.bitwise_and(adaptive_threshold, green_mask)

    cv2.imshow('frame', frame)
    cv2.imshow('s', green_mask)
    #cv2.imshow('e', segmented_image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np

# Callback function for the trackbar
def update_brightness(brightness_value):
    pass  # This function will be defined later, we'll use it to update the brightness of the camera feed

# Create a window to display the camera feed
cv2.namedWindow('Camera Feed')

# Create a trackbar to adjust brightness
cv2.createTrackbar('Brightness', 'Camera Feed', 0, 100, update_brightness)

# Open the default camera (usually camera index 0)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Get the current brightness value from the trackbar
    brightness_value = cv2.getTrackbarPos('Brightness', 'Camera Feed')

    # Add the brightness value to each pixel in the frame
    brighter_frame = cv2.add(frame, np.array([brightness_value]))

    # Display the frame
    cv2.imshow('Camera Feed', brighter_frame)

    # Check for key press and exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
