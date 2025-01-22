import cv2
import numpy as np
def detect_hexagon(contour):
    approx = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)
    return len(approx) == 6

image = cv2.imread("IMG_2264.webp")
pixels = image.reshape((-1, 3))
pixels = np.float32(pixels)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 7, 0.00002)
k = 3
_, labels, centers = cv2.kmeans(pixels, k, None, criteria, 7, cv2.KMEANS_USE_INITIAL_LABELS)
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(image.shape)
dst = cv2.GaussianBlur(segmented_image, (3, 3), 0)
edges = cv2.Canny(dst,100,200)
grey = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2LAB)
#contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#for contour in contours:
#    if detect_hexagon(contour):
#        M = cv2.moments(contour)
#        if M["m00"] != 0:
#            cx = int(M["m10"] / M["m00"])
#            cy = int(M["m01"] / M["m00"])
#            color = segmented_image[cy, cx]
#            cv2.drawContours(image, [contour], -1, tuple(map(int, color)), 2)

cv2.imshow('Image', dst)
cv2.imshow('Lolo', segmented_image)
cv2.imshow('sds', edges)
cv2.imshow('ds', image)
cv2.imshow('sdsd', grey)
cv2.waitKey(0)
cv2.destroyAllWindows()