import cv2
import dlib
import numpy as np
from imutils.video import VideoStream
import imutils
from sound import play, play2
#important landmark
LAND = [(0,0) for i in range(68)]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_68.dat')
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords
    
def detect_shape(img,outimg):
    global LAND
    ratio = len(outimg)/len(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale 
    rects = detector(gray, 1) # rects contains all the faces detected
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        for i in range(len(shape)):
            x,y = shape[i]
            LAND[i] = (x,y)
            if (i==66 or i==62):
                cv2.circle(outimg, (int(x*ratio), int(y*ratio)), 2, (0, 255, 0), -1)
            else:
                cv2.circle(outimg, (int(x*ratio), int(y*ratio)), 2, (0, 0, 255), -1)
    return img

vs = VideoStream(src=0).start()


i=0
# loop over the frames of the video
while True:
    i+=1
    frame = vs.read()
    nframe = imutils.resize(frame, width=400)
    img = detect_shape(nframe,frame)
    # cv2.imshow('Your beautiful face', frame)
    if i%3==0:
        play2(300+20*(LAND[66][1]-LAND[62][1]))
    key = cv2.waitKey(100) & 0xFF
    if key == ord("q"):
        break

# cleanup the camera and close any op*en windows
vs.stop()
cv2.destroyAllWindows()


cv2.destroyAllWindows()
