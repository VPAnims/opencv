import cv2
import numpy as np

purple = [224, 164, 166]
green = [37, 148, 74]
yellow = [68, 156, 234]
#close_purple
# You know what this is
def get_color_limits(color):
    BGR = np.uint8([[color]])
    HSV = cv2.cvtColor(BGR, cv2.COLOR_BGR2HSV)

    h = HSV[0][0][0]
    lower_limit = np.array([max(h - 20, 0), 100, 100], dtype=np.uint8)
    upper_limit = np.array([min(h + 20, 179), 255, 255], dtype=np.uint8)

    return lower_limit, upper_limit

# Lloyd's Algorithm
def group_colors(image, k):
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 30)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 20, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    clustered_image = centers[labels.flatten()]
    clustered_image = clustered_image.reshape(image.shape)

    return clustered_image, centers

# Color Display

# Closest Color Finder
def find_green(centers, specific_color):
    target_color = np.array(specific_color)
    min_distance = float('inf')
    closest_color = None
    closest_color_index = None

    for i, color in enumerate(centers):
        color_array = np.array(color)
        distance = np.linalg.norm(color_array - target_color)
        if distance < min_distance:
            min_distance = distance
            closest_color = color
            closest_color_index = i

    return closest_color

# Pixel Detector
def pixel_detector(image, final_image, green, yellow, purple):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        moments = cv2.moments(contour)
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        cv2.drawContours(final_image, [approx], 0, (0, 0, 0), 5)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        if len(approx) == 8:
            if moments["m00"] != 0:
                Cx = int(moments["m10"] / moments["m00"])
                Cy = int(moments["m01"] / moments["m00"])    
            else:
                Cy, Cx = 0, 0
            color1 = image[Cy, Cx]
            if ((color1[0] == green[0] and color1[1] == green[1] and color1[2] == green[2])):
                cv2.putText(final_image, "Place 2 Green Pixel here", (Cx, Cy), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
            if((color1[0] == yellow[0] and color1[1] == yellow[1] and color1[2] == yellow[2])):
                cv2.putText(final_image, "Place 2 Yellow Pixel here", (Cx, Cy), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
            if((color1[0] == purple[0] and color1[1] == purple[1] and color1[2] == purple[2])):
                cv2.putText(final_image, "Place 2 Purple Pixel here", (Cx, Cy), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
            cv2.circle(final_image, (Cx, Cy), 5, (255, 0, 0), -1)
        if len(approx) == 10:
            if moments["m00"] != 0:
                Cx = int(moments["m10"] / moments["m00"])
                Cy = int(moments["m01"] / moments["m00"])      
            else:
                Cy, Cx = 0, 0
            y1 = Cy+10
            y2 = Cy-10
            color1 = image[y1, Cx]  # Note the change in indexing
            color2 = image[y2, Cx]  # Note the change in indexing
            if ((color1[0] == green[0] and color1[1] == green[1] and color1[2] == green[2]) or (color1[0] == yellow[0] and color1[1] == yellow[1] and color1[2] == yellow[2])):
                if (color2[0] == green[0] and color2[1] == green[1] and color2[2] == green[2]):
                    print("Place Green Pixel")
                if(color2[0] == yellow[0] and color2[1] == yellow[1] and color2[2] == yellow[2]):
                    cv2.putText(image, "Place Purple Pixel here", (Cx, Cy), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            cv2.circle(image, (Cx, Cy), 5, (255, 0, 0), -1)


        if len(approx) == 12:
            if moments["m00"] != 0:
                cX = int(moments["m10"] / moments["m00"])
                cY = int(moments["m01"] / moments["m00"])
            else:
                cX, cY = 0, 0
            x1 = cX + 25
            x2 = cX - 25
            y1 = cY + 25
            color1 = image[cY, x1]
            color2 = image[cY, x2]
            color3 = image[y1, cX]
            print(x1, x2, y1)
            print(image[y1, cX], image[cY, x2], image[cY, x1])
            if (color1[0] == green[0] and color1[1] == green[1] and color1[2] == green[2]) or (color1[0] == purple[0] and color1[1] == purple[1] and color1[2] == purple[2]) or (color1[0] == yellow[0] and color1[1] == yellow[1] and color1[2] == yellow[2]):
                if (color2[0] == green[0] and color2[1] == green[1] and color2[2] == green[2]) or (color2[0] == purple[0] and color2[1] == purple[1] and color2[2] == purple[2]) or (color2[0] == yellow[0] and color2[1] == yellow[1] and color2[2] == yellow[2]):
                    if(color3[0] == green[0] and color3[1] == green[1] and color3[2] == green[2]) or (color3[0] == purple[0] and color3[1] == purple[1] and color3[2] == purple[2]) or (color3[0] == yellow[0] and color3[1] == yellow[1] and color3[2] == yellow[2]):
                        cv2.putText(final_image, "mosaic", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255))
            else:
                #cv2.putText(image, "mosaic", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0)) 
                print("not a mosaic")             
            cv2.circle(image, (cX, cY), 5, (255, 0, 0), -1)

    return image

# Image Processing
image = cv2.imread('sixth.jpg')
resize = cv2.resize(image, (1200, 900))

gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
image_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply adaptive thresholding
thresh = cv2.adaptiveThreshold(image_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 6)
color_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

# Add color to the thresholded image
color_img[thresh == 255] = resize[thresh == 255]
resized_image = cv2.bitwise_and(resize, color_img)

# Yellow Color
k = 7
clustered_image, y = group_colors(resized_image, k)
closest_yellow = find_green(y, yellow)
yellow_mask = cv2.inRange(clustered_image, closest_yellow, closest_yellow)
yellow_result = cv2.bitwise_and(clustered_image, clustered_image, mask=yellow_mask)

# Green Color
z = 10
green_image, g = group_colors(resized_image, z)
closest_green = find_green(g, green)
green_mask = cv2.inRange(green_image, closest_green, closest_green)
green_result = cv2.bitwise_and(green_image, green_image, mask=green_mask)

green_and_yellow = cv2.bitwise_or(green_result, yellow_result)

gray_image = cv2.cvtColor(green_and_yellow, cv2.COLOR_BGR2GRAY)
_, thresholded = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    mask = np.zeros(gray_image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -15, 255, thickness=cv2.FILLED)
    
    x, y, w, h = cv2.boundingRect(contour)
    
    for i in range(x, x + w):
        for j in range(y, y + h):
            if mask[j, i] == 255 and gray_image[j, i] == 0:  # If pixel is inside contour and black
                # Check neighboring pixels
                neighbors = [gray_image[j-1, i], gray_image[j+1, i], gray_image[j, i-1], gray_image[j, i+1]]
                if all(n != 2 for n in neighbors): 
                    green_and_yellow[j, i] = [green_and_yellow[j-1, i, 0], green_and_yellow[j-1, i, 1], green_and_yellow[j-1, i, 2]]

# Shape Detection
demo = pixel_detector(green_and_yellow, resized_image, closest_green,closest_yellow, purple)

# Displaying Results
cv2.imshow('Original Image', green_and_yellow)
cv2.imshow('sdf Image', resized_image)


cv2.waitKey(0)
cv2.destroyAllWindows()

