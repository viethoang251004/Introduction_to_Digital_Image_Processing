# Ex1
# Example 1
# importing OpenCV, time and Pandas library 
import cv2
import time
import pandas as pd
from datetime import datetime
# Assigning our static_back to None 
static_back = None
motion_list = [None, None]
time = []
df = pd.DataFrame(columns=["Start", "End"])
# Capturing video 
video = cv2.VideoCapture(0)
# Infinite while loop to treat stack of image as video 
while True:
    # Reading frame(image) from video 
    check, frame = video.read()
    motion = 0
    # Converting color image to grayscale image 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if static_back is None:
        static_back = gray
        continue
    diff_frame = cv2.absdiff(static_back, gray)
    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)
    cnts, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        motion = 1
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    motion_list.append(motion)
    motion_list = motion_list[-2:]
    if motion_list[-1] == 1 and motion_list[-2] == 0:
        time.append(datetime.now())
    if motion_list[-1] == 0 and motion_list[-2] == 1:
        time.append(datetime.now())
    cv2.imshow("Gray Frame", gray)
    cv2.imshow("Difference Frame", diff_frame)
    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Color Frame", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        if motion == 1:
            time.append(datetime.now())
        break
for i in range(0, len(time), 2):
    df = df.append({"Start": time[i], "End": time[i + 1]}, ignore_index=True)
df.to_csv("Time_of_movements.csv")
video.release()
cv2.destroyAllWindows()

# Example 2
import cv2
import numpy as np

video = cv2.VideoCapture('"D:/Hoang/NHAPMONXULYANHSO/THUCHANH/LAB_5/video.mp4"')
kernel = None
backgroundObject = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

while True:
    ret, frame = video.read()
    if not ret:
        break

    foreground_mask = backgroundObject.apply(frame)
    _, foreground_mask = cv2.threshold(foreground_mask, 250, 255, cv2.THRESH_BINARY)
    foreground_mask = cv2.erode(foreground_mask, kernel, iterations=1)
    foreground_mask = cv2.dilate(foreground_mask, kernel, iterations=2)

    contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frameCopy = frame.copy()

    for cnt in contours:
        if cv2.contourArea(cnt) > 400:
            x, y, width, height = cv2.boundingRect(cnt)
            cv2.rectangle(frameCopy, (x, y), (x + width, y + height), (0, 0, 255), 2)
            cv2.putText(frameCopy, 'Car Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1,
                        cv2.LINE_AA)

    foregroundPart = cv2.bitwise_and(frame, frame, mask=foreground_mask)

    stacked_frame = np.hstack((frame, foregroundPart, frameCopy))

    cv2.imshow('Original Frame, Extracted Foreground and Detected Cars',
               cv2.resize(stacked_frame, None, fx=0.5, fy=0.5))

    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()