import cv2
import numpy as np

def find_hexagon_with_inner_hexagon(image):
    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    adaptive_mask = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant)

# Apply the adaptive mask to the original image
    result_image = cv2.bitwise_and(image, image, mask=adaptive_mask)

    # Contour detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Contour filtering
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)
        if len(approx) == 6:  # Check if the contour is a hexagon
            # Hexagon validation
            # Calculate the area of the outer hexagon
            outer_area = cv2.contourArea(contour)
            
            # Find inner contours
            inner_contours, _ = cv2.findContours(contour, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            for inner_contour in inner_contours:
                inner_approx = cv2.approxPolyDP(inner_contour, 0.05 * cv2.arcLength(inner_contour, True), True)
                if len(inner_approx) == 6:  # Check if the contour is a hexagon
                    # Calculate the area of the inner hexagon
                    inner_area = cv2.contourArea(inner_contour)
                    
                    # If the inner hexagon's area is significantly smaller than the outer hexagon,
                    # consider it as the inner hexagon
                    if inner_area < 0.5 * outer_area:
                        # Color detection
                        rect = cv2.boundingRect(inner_contour)
                        x, y, w, h = rect
                        roi = image[y:y+h, x:x+w]
                        mean_color = cv2.mean(roi)
                        color = tuple(int(c) for c in mean_color[:3])
                        
                        # Draw contours and inner hexagon
                        cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)
                        cv2.drawContours(image, [inner_contour], 0, (0, 0, 255), 2)
                        
                        return color, image
                        
    return None, image

# Read the image
image = cv2.imread("hexagon_image.jpg")

# Find hexagon with inner hexagon and its color
color, result_image = find_hexagon_with_inner_hexagon(image)

if color is not None:
    print("Inner Hexagon Color:", color)
    cv2.imshow("Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Hexagon with inner hexagon not found.")