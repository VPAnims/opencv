import cv2
import numpy as np

cap = cv2.VideoCapture(0)
purple = [204, 50, 153]


def get_color_limits(color):
    BGR = np.uint8([[color]])
    HSV = cv2.cvtColor(BGR, cv2.COLOR_BGR2HSV)

    lower_limit = HSV[0][0][0] - 20, 100, 100
    upper_limit = HSV[0][0][0] + 40, 255, 255

    lower_limit = np.array(lower_limit, dtype=np.uint8)
    upper_limit = np.array(upper_limit, dtype=np.uint8)
    return lower_limit, upper_limit


while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    lower_limit, upper_limit = get_color_limits(color=purple)
    print(f"Lower limit: {lower_limit}")
    print(f"Upper limit: {upper_limit}")
    cv2.imshow('jsdks', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
