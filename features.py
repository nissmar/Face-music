import cv2
import dlib
import numpy as np
from imutils.video import VideoStream
import imutils


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_68.dat')
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords
    
def detect_shape(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale 
    rects = detector(gray, 1) # rects contains all the faces detected
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        for (x, y) in shape:
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
    return img

vs = VideoStream(src=0).start()


# loop over the frames of the video
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=200)
    img = detect_shape(vs.read())
    # left_eye, right_eye = detect_eyes(img)
    cv2.imshow('Your beautiful face', img)
    
    key = cv2.waitKey(100) & 0xFF
    if key == ord("q"):
        break

# cleanup the camera and close any op*en windows
vs.stop()
cv2.destroyAllWindows()


cv2.destroyAllWindows()
