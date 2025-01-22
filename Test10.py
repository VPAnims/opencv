import cv2
import numpy as np

'''def detect_colors(image, contours):
    colors = []
    for contour in contours:
        # Compute the centroid of the contour
        M = cv2.moments(contour)
        
        # Check if the area is non-zero before computing centroid
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Get the color of the centroid pixel
            color = image[cy, cx]
            colors.append(color)
        else:
            # Handle the case where the contour has zero area
            # You may choose to skip this contour or handle it differently
            print("Warning: Contour with zero area detected. Skipping.")
    return colors

def determine_next_color(colors):
    # Assuming colors are in BGR format
    colors = np.array(colors)
    unique_colors, counts = np.unique(colors, axis=0, return_counts=True)
    color_counts = dict(zip(map(tuple, unique_colors.tolist()), counts.tolist()))
    
    # Determine the next color based on the counts of adjacent colors
    # You can implement your specific logic here based on your rules
    
    # For demonstration, let's just return the color with the highest count
    next_color = max(color_counts, key=color_counts.get)
    return next_color

# Load the image
iL_mage = cv2.imread('Mosaics.jpg')
image = cv2.resize(iL_mage, (800, 600))
# Convert the image to grayscale

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply adaptive thresholding
thresh = cv2.adaptiveThreshold(image_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 6)

# Create a color image
color_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

# Add color to the thresholded image
color_img[thresh == 255] = image[thresh == 255]


# Display the image with detected squares
cv2.imshow('Detected Squares', image)
cv2.imshow('thresh', color_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np

imgld = cv2.imread('Mosaics.jpg', -1)
img = cv2.resize(imgld, (800, 600))
rgb_planes = cv2.split(img)

result_planes = []
result_norm_planes = []
for plane in rgb_planes:
    dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(plane, bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    result_planes.append(diff_img)
    result_norm_planes.append(norm_img)
    
result = cv2.merge(result_planes)
result_norm = cv2.merge(result_norm_planes)

cv2.imshow('shadows_out.png', result)
cv2.imshow('shadows_out_norm.png', result_norm)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np

# Read the input image
sds = cv2.imread('Mosaics.jpg')

image = cv2.resize(sds, (800, 600))

# Convert image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold to find black spots
_, thresholded = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY)

# Find contours of black spots
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through contours
for contour in contours:
    # Create a mask for the current contour
    mask = np.zeros(gray_image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    # Find bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Iterate through the bounding rectangle
    for i in range(x, x + w):
        for j in range(y, y + h):
            if mask[j, i] == 255 and gray_image[j, i] == 0:  # If pixel is inside contour and black
                # Check neighboring pixels
                neighbors = [gray_image[j-1, i], gray_image[j+1, i], gray_image[j, i-1], gray_image[j, i+1]]
                if all(n != 0 for n in neighbors):  # If all neighboring pixels are not black
                    # Fill black spot with surrounding color
                    image[j, i] = [image[j-1, i, 0], image[j-1, i, 1], image[j-1, i, 2]]

cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
import cv2
import numpy as np

# Define canvas size
width = 600
height = 400

# Create a black image
image = np.zeros((height, width, 3), dtype=np.uint8)

# Define colors
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
yellow = (0, 255, 255)

# Draw a rectangle
cv2.rectangle(image, (50, 50), (200, 150), red, -1)

# Draw a circle
cv2.circle(image, (300, 100), 50, green, -2)

# Draw a line
cv2.line(image, (100, 200), (400, 300), blue, 3)

# Draw filled triangle
pts = np.array([[150, 350], [250, 350], [200, 250]], dtype=np.int32)
cv2.fillConvexPoly(image, pts, yellow)

# Display the image
cv2.imshow("Colorful Shapes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''import cv2
import numpy as np

# Load your mask image (assuming it's a binary image)
mask = cv2.imread('your_mask_image.png', cv2.IMREAD_GRAYSCALE)

# Define a kernel for the closing operation
kernel = np.ones((5, 5), np.uint8)  # Adjust the kernel size as needed

# Perform closing to connect small disconnected components
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Convert all non-zero pixels (white) in the original mask to black
mask[closing == 255] = 0

# Save the modified mask
cv2.imwrite('modified_mask.png', mask)
'''