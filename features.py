import cv2
import dlib
import numpy as np
from imutils.video import VideoStream
import imutils
from sound import callback, set_freq, set_mix
import sounddevice as sd
from tensorflow import keras
from pickle import load

from landmark_pickle import LandmarkSaver, load_landmark, display_time
from explicitely_killing_hugo import np_to_complex, mean_ratio, lat_angle,vert_angle
from landmark_processing import normalize_landmark

#tensorflow model
dnn_model = keras.models.load_model('mouth_model')
#SVM model
with open('mouth_svm.pickle','rb') as pickle_in:
    svm_regr = load(pickle_in)
with open('tilt_svm.pickle','rb') as pickle_in:
    svm_tilt = load(pickle_in)
with open('pan_svm.pickle','rb') as pickle_in:
    svm_pan = load(pickle_in)


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
            if (i in [7,8,9,21,22,31,35]):
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



def evaluate_nn(landmark):
    if landmark is None:
        return 0
    nn_input = np.array([normalize_landmark(landmark)])
    return dnn_model.predict(nn_input)[0]

def evaluate_svm(landmark):
    if landmark is None:
        return 0
    input = np.array([normalize_landmark(landmark)])
    return svm_regr.predict(input)[0]


def evaluate_svm_pose(landmark):
    if landmark is None:
        return 0
    input = np.array([normalize_landmark(landmark)])
    return svm_tilt.predict(input)[0], svm_pan.predict(input)[0]


i=0


ready_for_record = False
is_capturing = False
stream = sd.OutputStream(channels=1, callback=callback,blocksize=1500)


### KEY LANDMARK INTERPOLATION
stream.start()

## custom landmark
l1,l2 = None,None
l3,l4 = None,None

## personal calibration
cal = []
calibrated = False
cal_v0,cal_v1 = None,None
cal_l0,cal_l1 = None,None

while True:
    """ press space two times to define the first and second references. Press space again to erase """
    i+=1
    frame = vs.read()
    frame = cv2.flip(frame,1)
    nframe = imutils.resize(frame, width=400)
    landmark = detect_shape(nframe,frame)
    if not(landmark is None):
        if (calibrated):
                r = vert_angle(np_to_complex(landmark),cal_v0,cal_v1)
                r = min(1,max(0,r))
                
                r2 = lat_angle(np_to_complex(landmark),cal_l0,cal_l1)
                r2 = min(max(r2,0),1)
                set_freq(440*(1+0.5*r2))
                set_mix(r)
                frame = display_time(frame,r)
                frame = display_time(frame,min(max(r2,0),1),300)
                height, width, _ = frame.shape

                cv2.circle(frame, (int(r2*width), int(r*height)), 10, (0, 255, 0), -1)
        else:
            cal.append([vert_angle(np_to_complex(landmark),0,1),lat_angle(np_to_complex(landmark),0,1)])

        if not(l2 is None):
            r = mean_ratio(l1,l2,np_to_complex(landmark))
            r = min(max(r,0),1)
            set_freq(440*(1+0.5*r))
            frame = display_time(frame,r)
            if not(l4 is None):
                r = mean_ratio(l3,l4,np_to_complex(landmark))
                r = min(max(r,0),1)
                set_mix(r)
                frame = display_time(frame,min(max(r,0),1),300)
        else:
            r = evaluate_svm(landmark)
            if r<0.3:
                tresh = 20
                r,r2 = evaluate_svm_pose(landmark)
                r+=tresh
                r/=2*tresh
                r = min(1,max(0,r))
                r2+=tresh
                r2/=2*tresh
                r2 = min(1,max(0,r2))
                r,r2 = 1-r,1-r2


                dist = [abs(r2),abs(r2-0.5), abs(1-r2)]
                dist2 = [abs(r),abs(1-r)]
                mult = [147,196,220,330,294,261][dist.index(min(dist))+3*dist2.index(min(dist2))]
                set_mix(0.3)
                set_freq(mult)
                frame = display_time(frame,r2,300)
                frame = display_time(frame,r)
                height, width, _ = frame.shape
                cv2.circle(frame, (int(r2*width), int(r*height)), 10, (0, 255, 0), -1)
                



    cv2.imshow('Your beautiful face', frame)

    key = cv2.waitKey(100) & 0xFF
    if key == ord("a"): # calibration ok
        if (calibrated):
            calibrated = False
            cal = []
        else:
            v_cal = [e[0] for e in cal]
            l_cal = [e[1] for e in cal]
            cal_v0, cal_v1 = min(v_cal), max(v_cal)
            cal_l0, cal_l1 = min(l_cal), max(l_cal)
            calibrated = True
    if key == ord("q"):
        break
    if key == ord("n"): #show neural network estimation
        x = evaluate_nn(landmark)[0]
        print(x)
        x = min(1,max(0,x))
        set_freq(440*(1+0.5*x))
    if key == ord('s'):
        x = evaluate_svm(landmark)
        print(x)
    if ready_for_record and not is_capturing and key == ord(" "):
        print("Starting to record for emotion:", saver.emotion_label)
        ready_for_record = False
        is_capturing = True
        saver.begin_record()
    if not ready_for_record and not is_capturing and key == ord(" "):
        ready_for_record = True
        print("Please type an emotion label corresponding to this record:")
        emotion_label = input()
        saver.set_emotion_label(emotion_label)
    
    if key == ord("c"):
        if not(landmark is None):
            if l1 is None:
                l1 = np_to_complex(landmark)
            elif l2 is None:
                l2 = np_to_complex(landmark)
            elif l3 is None:
                l3 = np_to_complex(landmark)
            elif l4 is None:
                l4 = np_to_complex(landmark)
    if key == ord("e"):
        l1,l2 = None,None
        l3,l4 = None,None
   
stream.stop()
# cleanup the camera and close any open windows
vs.stop()
cv2.destroyAllWindows()