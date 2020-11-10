import cv2
import dlib
import numpy as np
from imutils.video import VideoStream
import imutils
from sound import callback, set_freq
import sounddevice as sd

from landmark_pickle import LandmarkSaver, load_landmark, display_time
from explicitely_killing_hugo import np_to_complex, mean_ratio

saver = LandmarkSaver()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_68.dat')

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords
    
def detect_shape(img,outimg):
    landmark = None
    ratio = len(outimg)/len(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale 
    rects = detector(gray, 1) # rects contains all the faces detected
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        landmark = shape_to_np(shape)
        for i in range(len(landmark)):
            x,y = landmark[i]
            if (i in [39,42]):
                cv2.circle(outimg, (int(x*ratio), int(y*ratio)), 2, (0, 255, 0), -1)
            else:
                cv2.circle(outimg, (int(x*ratio), int(y*ratio)), 2, (0, 0, 255), -1)
    return landmark

vs = VideoStream(src=0).start()

def face_frequency(landmark):
    mouse_open = (landmark[66][1]-landmark[62][1])
    fac = 1

    # print((LAND[40][1]+LAND[41][1]-LAND[38][1]-LAND[37][1]), " vs ", (LAND[39][0]-LAND[36][0]))
    # if (LAND[40][1]+LAND[41][1]-LAND[38][1]-LAND[37][1])<0.43*(LAND[39][0]-LAND[36][0]):
    #     fac = 1.5
    return (mouse_open*20 + 300)


i=0


ready_for_record = False
is_capturing = False
stream = sd.OutputStream(channels=1, callback=callback,blocksize=2000)

### CAPTURING LANDMARKS
# stream.start()
# while True:
#     i+=1
#     frame = vs.read()
#     nframe = imutils.resize(frame, width=400)
#     landmark = detect_shape(nframe,frame)

#     set_freq(face_frequency(landmark))

#     if is_capturing:
#         is_capturing, frame = saver.poke(frame, landmark)
#     cv2.imshow('Your beautiful face', frame)

#     key = cv2.waitKey(100) & 0xFF
#     if key == ord("q"):
#         break
#     if ready_for_record and not is_capturing and key == ord(" "):
#         print("Starting to record for emotion:", saver.emotion_label)
#         ready_for_record = False
#         is_capturing = True
#         saver.begin_record()
#     if not ready_for_record and not is_capturing and key == ord(" "):
#         ready_for_record = True
#         print("Please type an emotion label corresponding to this record:")
#         emotion_label = input()
#         saver.set_emotion_label(emotion_label)

### 

l1,l2 = None,None
while True:
    i+=1
    frame = vs.read()
    nframe = imutils.resize(frame, width=400)
    landmark = detect_shape(nframe,frame)
    if not(l2 is None or landmark is None):
        r = mean_ratio(l1,l2,np_to_complex(landmark))
        frame = display_time(frame,min(max(r,0),1))

    cv2.imshow('Your beautiful face', frame)

    key = cv2.waitKey(100) & 0xFF
    if key == ord("q"):
        break
    if key == ord(" "):
        if l1 is None:
            l1 = np_to_complex(landmark)
        elif l2 is None:
            l2 = np_to_complex(landmark)
        else:
            l1,l2 = None, None
   
stream.stop()
# cleanup the camera and close any open windows
vs.stop()
cv2.destroyAllWindows()