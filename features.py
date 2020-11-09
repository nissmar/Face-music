import cv2
import dlib
import numpy as np
from imutils.video import VideoStream
import imutils
from sound import callback, set_freq
import sounddevice as sd

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
            if (i==66 or i==62 or i==38 or i ==37 or i ==40 or i ==41 ):
                cv2.circle(outimg, (int(x*ratio), int(y*ratio)), 2, (0, 255, 0), -1)
            else:
                cv2.circle(outimg, (int(x*ratio), int(y*ratio)), 2, (0, 0, 255), -1)
    return img

vs = VideoStream(src=0).start()


def face_frequency():
    mouse_open = (LAND[66][1]-LAND[62][1])
    fac = 1

    # print((LAND[40][1]+LAND[41][1]-LAND[38][1]-LAND[37][1]), " vs ", (LAND[39][0]-LAND[36][0]))
    # if (LAND[40][1]+LAND[41][1]-LAND[38][1]-LAND[37][1])<0.43*(LAND[39][0]-LAND[36][0]):
    #     fac = 1.5
    return (mouse_open*20 + 300)*fac


i=0
stream = sd.OutputStream(channels=1, callback=callback,blocksize=2000)
stream.start()

# loop over the frames of the video
while True:
    i+=1
    frame = vs.read()
    nframe = imutils.resize(frame, width=400)
    img = detect_shape(nframe,frame)
    frame = cv2.flip(frame, 1)
    # cv2.imshow('Your beautiful face', frame)
    set_freq(face_frequency())
    key = cv2.waitKey(100) & 0xFF
    if key == ord("q"):
        break
stream.stop()

# cleanup the camera and close any open windows
vs.stop()
cv2.destroyAllWindows()


cv2.destroyAllWindows()
