import cv2
import numpy as np

cap = cv2.VideoCapture(0)

Green = [0, 128, 0]

def find_hexagon_with_inner_hexagon(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)
        if len(approx) == 6:  # Check if the contour is a hexagon
            outer_area = cv2.contourArea(contour)
            
            inner_contours, _ = cv2.findContours(contour, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            for inner_contour in inner_contours:
                inner_approx = cv2.approxPolyDP(inner_contour, 0.05 * cv2.arcLength(inner_contour, True), True)
                if len(inner_approx) == 6:  # Check if the contour is a hexagon
                    inner_area = cv2.contourArea(inner_contour)
                    

                    if inner_area < 0.5 * outer_area:
                        # Color detection
                        rect = cv2.boundingRect(inner_contour)
                        x, y, w, h = rect
                        roi = image[y:y+h, x:x+w]
                        mean_color = cv2.mean(roi)
                        color = tuple(int(c) for c in mean_color[:3])
                        
                        cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)
                        cv2.drawContours(image, [inner_contour], 0, (0, 0, 255), 2)
                        
                        return color, image
                        
    return None, image

while True: 
    ret, frame = cap.read()

    color, result_image = find_hexagon_with_inner_hexagon(frame)


    cv2.imshow('e', frame)

    if color is not None:
        print("Inner Hexagon Color:", color)
        cv2.imshow("Result", result_image)
    else:
        print("Hexagon with inner hexagon not found.")
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()