import cv2
import numpy as np

cap = cv2.VideoCapture(0)

green = [0, 128, 0]

def get_color_limits(color):
    BGR = np.uint8([[color]])
    HSV = cv2.cvtColor(BGR, cv2.COLOR_BGR2HSV)

    lower_limit = HSV[0][0][0] - 40, 100, 100
    upper_limit = HSV[0][0][0] + 40, 255, 255

    lower_limit = np.array(lower_limit, dtype=np.uint8)
    upper_limit = np.array(upper_limit, dtype=np.uint8)
    
while True: 
    ret, frame = cap.read()

    pixels = frame.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 7, 0.00002)
    k = 6
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 7, cv2.KMEANS_USE_INITIAL_LABELS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(frame.shape)

    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(segmented_image,-1,kernel)
    #blurred = cv2.GaussianBlur(segmented_image, (1, 1), 0)
    #edges = cv2.Canny(blurred,100,200)

    cv2.imshow('frame', dst)
    #cv2.imshow('s', edges)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()