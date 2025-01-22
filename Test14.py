import cv2
import numpy as np

purple = [224, 164, 166]
green  = [73,126,60]
yellow = [68,156,234]

#You know what this is
def get_color_limits(color):
    BGR = np.uint8([[color]])
    HSV = cv2.cvtColor(BGR, cv2.COLOR_BGR2HSV)

    h = HSV[0][0][0]
    lower_limit = np.array([max(h - 20, 0), 100, 100], dtype=np.uint8)
    upper_limit = np.array([min(h + 20, 179), 255, 255], dtype=np.uint8)

    return lower_limit, upper_limit

#Lloyds Algorithm
def group_colors(image, k):
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    clustered_image = centers[labels.flatten()]
    clustered_image = clustered_image.reshape(image.shape)

    return clustered_image, centers

#Color Display
def display_colors(centers):
    color_display = np.zeros((100, len(centers) * 100, 3), dtype=np.uint8)
    for i, color in enumerate(centers):
        print(f"Cluster {i+1}: BGR({color[0]}, {color[1]}, {color[2]})")

    for i, color in enumerate(centers):
        color_display[:, i * 100:(i + 1) * 100] = color

    cv2.imshow('Clustered Colors', color_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Closest Color Finder
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
    
    return closest_color, closest_color_index

#Pixel Detector
def pixel_detector(image, final_image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grey, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        cv2.drawContours(final_image, [approx], 0, (166, 164, 224), 5)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        if len(approx) == 6:
            cv2.putText(final_image, "Pixel", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0))
    
    return image
#image
image = cv2.imread('X.jpg') 
resized_image = cv2.resize(image, (640, 320))
k = 6
clustered_image, centers = group_colors(resized_image, k)
closest_color, closest_color_index = find_green(centers, yellow)

#Color Detection
hsv = cv2.cvtColor(clustered_image, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(clustered_image, closest_color, closest_color)
result = cv2.bitwise_and(clustered_image, clustered_image, mask = mask)
print(f"Closest color to green: Cluster {closest_color_index + 1} - BGR({closest_color[0]}, {closest_color[1]}, {closest_color[2]})")

grey = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

#Shape Detection
Demo = pixel_detector(result, resized_image)
#result
cv2.imshow('Pop-popular, born to be popular She in debt, 20 mill, but she run it up She can never be broke cause she popular Turn the webcam on for the followers beggin on her knees to be popular Thats her dream, to be popular (hey) Kill anyone to be popularSell her soul to be popular ', resized_image)
cv2.imshow('Clustered Image', result)
cv2.imshow('Original Image', clustered_image)
cv2.imshow('HSV', hsv)
cv2.imshow('Mask', grey)
cv2.imshow('sd', Demo)
display_colors(centers)
