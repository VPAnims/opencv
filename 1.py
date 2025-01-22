import cv2
import numpy as np
'''import os

cap = cv2.VideoCapture(0)
yellow = [0, 255, 255]
green = [0, 128, 0]
purple = [204, 50, 153]

def get_color_limits(color):
    BGR = np.uint8([[color]])
    HSV = cv2.cvtColor(BGR, cv2.COLOR_BGR2HSV)

    lower_limit = HSV[0][0][0] - 20, 100, 100
    upper_limit = HSV[0][0][0] + 20, 255, 255

    lower_limit = np.array(lower_limit, dtype=np.uint8)
    upper_limit = np.array(upper_limit, dtype=np.uint8)
    return lower_limit, upper_limit

image = cv2.imread("Green.jpg")

lower_Green, upper_Green = get_color_limits(green)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
green_mask = cv2.inRange(hsv, lower_Green, upper_Green)
green_result = cv2.bitwise_and(image, image, mask=green_mask)

cv2.imshow('e', image)
cv2.imshow('s', green_mask)
cv2.imshow('l', green_result)
def find(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
arr1 = np.array([1, 2, 3])
arr2 = np.array([7, 12, 10])

# Calculate the difference between the two arrays
diff = arr1 - arr2

# Find the index of the smallest difference
idx = np.argmin(diff)

# Print the closest array
print(idx)'''
import cv2
import time

# Open the video stream
cap = cv2.VideoCapture(0)  # You can also use '0' for webcam or another video device

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

# Initialize variables
start_time = time.time()
capture_interval = 10  # in seconds

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from video stream")
        break

    # Check if 10 seconds have elapsed
    elapsed_time = time.time() - start_time
    if elapsed_time >= capture_interval:
        # Save the frame as an image
        cv2.imwrite('captured_frame.jpg', frame)
        print("Frame captured at {} seconds".format(int(elapsed_time)))

        # Display the captured image
        cv2.imshow('Captured Frame', frame)
        cv2.waitKey(0)  # Wait indefinitely until a key is pressed or use a delay in milliseconds
        
        start_time = time.time()  # Reset start time for next capture

    # Check if the 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close windows
cap.release()
cv2.destroyAllWindows()
