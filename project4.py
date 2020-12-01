# import face_recognition
import cv2 as cv
import os
import numpy as np


video_capture = cv.VideoCapture(0)
path = '../'
while video_capture.isOpened():
    ret, img = video_capture.read()

    if ret:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # face_location = face_recognition.face_locations(gray)
        # for (top,right,bottom,left) in face_location:
        #     cv.rectangle(gray,(left,top),(right,bottom),(0,0,255),2)
        #     cv.rectangle(gray, (left, bottom - 35), (right, bottom), (0, 0, 255))
        #     font = cv.FONT_HERSHEY_DUPLEX
        #     cv.putText(gray,"human", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv.imshow('video',gray)

    if cv.waitKey(3) ==27:
        cv.imwrite(path,gray)
        break

cv.destroyAllWindows()
