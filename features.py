import cv2
import dlib
import numpy as np
from imutils.video import VideoStream
import imutils

from landmark_saver import LandmarkSaver


saver = LandmarkSaver()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_68.dat')
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords
    
def detect_shape(img):
    landmark = {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale 
    rects = detector(gray, 1) # rects contains all the faces detected
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        for i in range(len(shape)):
            x,y = shape[i]
            landmark[i] = (x,y)
            if (i==66 or i==62):
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
            else:
                cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
    #print(landmark[66]-landmark[62])
    return img, landmark

vs = VideoStream(src=-1).start()
is_capturing = False

# loop over the frames of the video
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    img, landmark = detect_shape(frame)
    # left_eye, right_eye = detect_eyes(img)
    img = cv2.resize(img,(960,540))
    cv2.imshow('Your beautiful face', img)

    if is_capturing:
        is_capturing, img = saver.poke(img, landmark)
    
    key = cv2.waitKey(100) & 0xFF
    if key == ord("q"):
        break
    if not is_capturing and key == ord(" "):
        is_capturing = True
        saver.begin_capture()

# cleanup the camera and close any op*en windows
vs.stop()
cv2.destroyAllWindows()


cv2.destroyAllWindows()
