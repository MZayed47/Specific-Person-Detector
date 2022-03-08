import os
import re
import cv2
import numpy as np
from glob import glob
import shutil

import time
from time import gmtime, strftime
from datetime import datetime
import tensorflow as tf

import face_recognition
import csv


# Source Images
images = []
classNames = []


yy = '03-Mar-2022_07-14'

path = 'People'
cl = "Mashrukh Zayed" + ".jpg"
curImg = cv2.imread(f'{path}/{cl}')
images.append(curImg)
classNames.append(os.path.splitext(cl)[0])

# Face Encodings
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)

file = "./detections/crop_" + yy + "/"

for i in os.listdir(file):
    img_name = i.split('.')[0]
    # print(img_name)
    try:
        frame = cv2.imread("./detections/crop_" + yy + "/" + i)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(frame)
        encodesCurFrame = face_recognition.face_encodings(frame, facesCurFrame)

        person_path = os.path.join(os.getcwd(), 'detections', 'person_' + yy)
        isdir = os.path.isdir(person_path)
        if not isdir:
            os.mkdir(person_path)

        for encodeFace,faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name, 'is present in the video.\n')

                y1,x2,y2,x1 = faceLoc
                y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(frame, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255),2)

                cv2.imwrite(person_path + '/' + img_name + '_' + name + '.jpg', frame)

    except:
        pass

# cv2.destroyAllWindows()

