import mediapipe as mp
import cv2
import numpy as np

file = "src/push-up.mp4"

cap = cv2.VideoCapture(file)

while cap.isOpened():
    ret, frame = cap.read()

    if ret == True:
        cv2.imshow("Frame", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()