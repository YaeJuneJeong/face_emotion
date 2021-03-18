import cv2 as cv
# Using Face_recoginition
import face_recognition
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import time

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

classifier = load_model('../../Desktop/python/acc_0.54.h5')
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Scared']

cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    labels = []
    if ret:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_recognition.face_locations(img)

        for (top, right, bottom, left) in faces:
            cv.rectangle(gray, (left, top), (right, bottom), (255, 0, 0), 2)
            roi_gray = gray[top:right, top: bottom]

            roi_gray = cv.resize(roi_gray, (48, 48), interpolation=cv.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = classifier.predict(roi)[0]
                label = class_labels[preds.argmax()]
                cv.putText(gray, label, (top - 6, right + 6), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)
    cv.imshow('face', gray)
    # time.sleep(0.1)

    if cv.waitKey(3) == 27:
        break
cv.destroyAllWindows()
