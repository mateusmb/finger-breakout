import cv2
import numpy as np

def ThresholdFrame(frame):
    hsvim = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    skinRegionHSV = cv2.inRange(hsvim, lower, upper)
    blurred = cv2.blur(skinRegionHSV, (2,2))
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)
    return thresh

def ContoursFrame(frame, thresh):
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    cv2.drawContours(frame, [contours], -1, (255,255,0), 2)
    return frame


def main():
    cap = cv2.VideoCapture(0)

    while(True):
        _, frame = cap.read()

        thresh = ThresholdFrame(frame)
        frame = ContoursFrame(frame, thresh)

        cv2.imshow('Finger Detecion', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()