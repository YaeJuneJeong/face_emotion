import face_recognition
import cv2 as cv
import os
import numpy as np
import glob

video_capture = cv.VideoCapture(0)

target = []
known = []

for filename in glob.glob('C:/Users/jyj98/Desktop/project/table/*.jpg'):
    picture = face_recognition.load_image_file(filename)
    name = filename.split('\\',-1)
    name = name[-1].split('.',2)
    target.append(name[0])
    trait = face_recognition.face_encodings(picture)[0]
    known.append(trait)

print(target)
while video_capture.isOpened():
    ret, img = video_capture.read()


    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = img[:, :, ::-1]
    if ret:
        # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        face_location = face_recognition.face_locations(rgb_small_frame,model='cnn')
        my_face_incodings = face_recognition.face_encodings(rgb_small_frame, face_location)
        man_list = []
        for face_incoding in my_face_incodings:
            face_recognition.compare_faces(known, face_incoding,tolerance=0.5)
            face_distance = face_recognition.face_distance(known, face_incoding)
            name = "???" if np.argmin(face_distance) > 0.4 else target[np.argmin(face_distance)]
            man_list.append(name)
        for (top, right, bottom, left), name in zip(face_location,man_list):
            cv.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv.FONT_HERSHEY_DUPLEX
            cv.putText(img, name, (left, top), font, 1.0, (255, 255, 255), 1)

        cv.putText(img, 'human population : '+str(len(face_location)) if face_location is not None else str(0), (0, 50), cv.FONT_HERSHEY_DUPLEX, 1.0,(0, 0, 0), 1)

    cv.imshow('video', img)

    if cv.waitKey(3) == 27:
        # cv.imwrite(path,img)
        break

cv.destroyAllWindows()
