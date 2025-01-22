import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def group_colors(image, k):
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 30)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 20, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    clustered_image = centers[labels.flatten()]
    clustered_image = clustered_image.reshape(image.shape)
    return clustered_image, centers

def find_similar_colors(image, target_color, color_range=40):
    # Convert the image to HSV (Hue, Saturation, Value) color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Convert target RGB color to HSV
    target_hsv = cv2.cvtColor(np.uint8([[target_color]]), cv2.COLOR_BGR2HSV)[0][0]

    # Define a lower and upper bound for the color range in HSV
    lower_bound = np.array([target_hsv[0] - color_range, target_hsv[1] - color_range, target_hsv[2] - color_range])
    upper_bound = np.array([target_hsv[0] + color_range, target_hsv[1] + color_range, target_hsv[2] + color_range])

    # Mask for the color range
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Convert the image to grayscale before applying adaptive threshold
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive threshold to the grayscale image
    block_size = 11
    constant = 2
    adaptive_thresholdMask = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant)

    # Combine the color mask with the adaptive threshold mask
    mask2 = cv2.bitwise_or(mask, adaptive_thresholdMask)

    # Find the pixels within the color range
    result = cv2.bitwise_and(image, image, mask=mask2)

    return result


while True:
    ret, frame = cap.read()
    if not ret:
        break

    lower_blue = np.array([120, 61, 9], dtype=np.uint8)
    upper_blue = np.array([130, 67, 16], dtype=np.uint8)

    lower_red = np.array([53, 30, 200], dtype=np.uint8)
    upper_red = np.array([75, 47, 225], dtype=np.uint8)


    height, width, _ = frame.shape
    cx = int(width / 2)
    cy = int(height / 2)

    clustered_image, centers = group_colors(frame, 5)
 
    # Blur the clustered image
    clustered_image_blurred = cv2.GaussianBlur(clustered_image, (15, 15), 0)

    pixel_center = clustered_image_blurred[cy, cx]
    print(f"Center Pixel Color: {pixel_center}")
    result = find_similar_colors(clustered_image, [140, 90, 40])

    cluster_gray = cv2.cvtColor(clustered_image_blurred, cv2.COLOR_BGR2GRAY)
    adaptive_thresholdMask = cv2.adaptiveThreshold(cluster_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    mask = cv2.inRange(clustered_image_blurred, lower_blue, upper_blue)

    new_mask = cv2.bitwise_or(adaptive_thresholdMask, mask)
    # Create masks
    mask = cv2.inRange(clustered_image_blurred, lower_blue, upper_blue)
    redmask = cv2.inRange(clustered_image_blurred, lower_red, upper_red)

    # Blur the masks
    mask_blurred = cv2.GaussianBlur(mask, (15, 15), 0)
    redmask_blurred = cv2.GaussianBlur(redmask, (15, 15), 0)

    # Combine the blurred masks
    combined_mask = cv2.bitwise_or(mask_blurred, redmask_blurred)

    # Apply the combined blurred mask to the blurred clustered image
    resul3t = cv2.bitwise_and(clustered_image_blurred, clustered_image_blurred, mask=new_mask)

    # Draw a circle at the center of the frame
    cv2.circle(clustered_image_blurred, (cx, cy), 5, [255, 0, 0], 3)

    # Display the results
    cv2.imshow("Blurred Clustered Frame", clustered_image_blurred)
    cv2.imshow('adap', adaptive_thresholdMask)
    cv2.imshow("Final Result", resul3t)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
