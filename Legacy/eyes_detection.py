from imutils.video import VideoStream
import imutils
import numpy as np
import cv2
import datetime
from time import time

from trainer import Trainer


trainer = Trainer()
vs = VideoStream(src=0).start()
#face_cascade = cv2.CascadeClassifier(''/root/opencv/data/haarcascades/haarcascade_frontalface_default.xml'')
eye_cascade = cv2.CascadeClassifier('eyes_haar_features.xml')

def sample(img, rectangle):
    ex,ey,ew,eh = rectangle
    return img[ey:ey+eh, ex:ex+ew]

def detect_eyes(img):
    t1 = time()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # for (x,y,w,h) in faces:
    #      cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #      roi_gray = gray[y:y+h, x:x+w]
    #      roi_color = img[y:y+h, x:x+w]
    #      eyes = eye_cascade.detectMultiScale(roi_gray)
    #      for (ex,ey,ew,eh) in eyes:
    #          cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


    draft_detection,confidence = eye_cascade.detectMultiScale2(gray)
    confidence = [x[0] for x in confidence]
    sorted_by_confidence = sorted(list(zip(draft_detection,confidence)), key = lambda x: -x[1])
    eyes = [x[0] for x in sorted_by_confidence[:2]]

    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    # print(time()-t1)
    # print(len(eyes))

    if len(eyes) == 2:
        if eyes[0][0] > eyes[1][0]:
            left_eye = eyes[0]
            right_eye = eyes[1]
        else:
            left_eye = eyes[1]
            right_eye = eyes[0]
        return cv2.resize(sample(img, left_eye), (100,100)), cv2.resize(sample(img, right_eye), (100,100))

    return np.zeros((100,100,3),np.uint8), np.zeros((100,100,3),np.uint8)



# loop over the frames of the video
while True:
    img = vs.read()
    left_eye, right_eye = detect_eyes(img)
    cv2.imshow('Your beautiful face', img)
    cv2.imshow('left eye', left_eye)
    cv2.imshow('right eye', right_eye)
    key = cv2.waitKey(100) & 0xFF
    if key == ord("q"):
        break
    if key == ord("t"):
        trainer.train(left_eye,right_eye)
# cleanup the camera and close any op*en windows
vs.stop()
cv2.destroyAllWindows()


cv2.destroyAllWindows()
