import cv2
import dlib
import numpy as np
from imutils.video import VideoStream
import imutils
from sound import callback, set_freq, set_mix,set_harmonic, get_wave
import sounddevice as sd
from tensorflow import keras
from pickle import load

from landmark_pickle import load_landmark, display_time
from explicit import np_to_complex, mean_ratio, lat_angle,vert_angle, normalized_to_complex, landmark_angle,rotate_landmark
from landmark_processing import normalize_landmark
import matplotlib.pyplot as plt
#tensorflow model
dnn_model = keras.models.load_model('mouth_model')
#SVM model
with open('mouth_svm.pickle','rb') as pickle_in:
    svm_mouth = load(pickle_in)
with open('sourcils_svm.pickle','rb') as pickle_in:
    svm_eyebrows = load(pickle_in)
with open('tilt_svm.pickle','rb') as pickle_in:
    svm_tilt = load(pickle_in)
with open('pan_svm.pickle','rb') as pickle_in:
    svm_pan = load(pickle_in)


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



def evaluate_nn(normalized_landmark):
    if normalized_landmark is None:
        return 0
    nn_input = np.array([normalized_landmark])
    return dnn_model.predict(nn_input)[0]


def evaluate_svm(normalized_landmark, regr):
    if landmark is None:
        return 0
    input = np.array([normalized_landmark])
    return regr.predict(input)[0]


def evaluate_svm_pose(normalized_landmark):
    if normalized_landmark is None:
        return 0
    input = np.array([normalized_landmark])
    return svm_tilt.predict(input)[0], svm_pan.predict(input)[0]


i=0


ready_for_record = False
is_capturing = False


# Sound
stream = sd.OutputStream(channels=1, callback=callback,blocksize=1500)
set_harmonic(0)
set_mix(0.2)
stream.start()

## eyebrows
l1,l2 = None,None

def draw_complex(land,img):
    for e in land:
        cv2.circle(img, (int(100*e.real), int(100*e.imag)), 2, (0, 255, 0), -1)
while True:
    """ press space two times to define the first and second references. Press space again to erase """
    i+=1
    frame = vs.read()
    frame = cv2.flip(frame,1)
    nframe = imutils.resize(frame, width=400)
    landmark = detect_shape(nframe,frame)
    if not(landmark is None):
        complex_landmark = np_to_complex(landmark)
        angle = landmark_angle(complex_landmark)
        normalized_landmark = normalize_landmark(rotate_landmark(complex_landmark,-angle))
        height, width, _ = frame.shape

        cv2.line(frame,(int(width/3), 0),(int(width/3), int(height)), (255, 0, 0))
        cv2.line(frame,(int(2*width/3), 0),(int(2*width/3), int(height)), (255, 0, 0))
        cv2.line(frame,(0,int(height/2)),(int(width), int(height/2)), (255, 0, 0))
        tresh = 20
        r,r2 = evaluate_svm_pose(normalized_landmark)
        r+=tresh
        r/=2*tresh
        r = min(1,max(0,r))
        r2+=tresh
        r2/=2*tresh
        r2 = min(1,max(0,r2))
        r,r2 = 1-r,1-r2


        dist = [abs(r2-1.0/6),abs(r2-0.5), abs(1-1.0/6-r2)]
        dist2 = [abs(r),abs(1-r)]
        index = dist.index(min(dist))+3*dist2.index(min(dist2))
        # mult = [147,196,220,330,294,261][index] #pentatonic
        mult = [261.63,196.00,164.81,329.63,220.00,174.61][index] #pentatonic
        print(angle)
        if not(l2 is None): #the eyebrows have been calibrated
            brow = mean_ratio(l1,l2,complex_landmark)
            brow = min(max(brow,0),1)
            frame = display_time(frame,brow)
            set_harmonic(brow)
            # if (brow<0.4):
            #     print("bop")
            #     set_mix(0.2)
            # else:
            #     set_mix(0.7)
            
        if evaluate_svm(normalized_landmark,svm_mouth)<0.3:
            set_freq(mult)
            cv2.circle(frame, (int(r2*width), int(r*height)), 10, (0, 255, 0), -1)
        else: # the mouth is open
            cv2.circle(frame, (int(r2*width), int(r*height)), 10, (0, 255, 255), -1)



    cv2.imshow('Face music', frame)

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
        wave = get_wave()
        print(len(wave))
        # plt.plot(np.concatenate(wave[-10:]))
        break
    if key == ord('s'): # evaluate landmark with mouth and eyebrows svm
        x = evaluate_svm(normalize_landmark, svm_mouth)
        print("Mouth:", x)
        x = evaluate_svm(normalize_landmark, svm_eyebrows)
        print("Eyebrows:", x)
    if key == ord("c"):
        if not(landmark is None):
            if l1 is None:
                l1 = complex_landmark
            elif l2 is None:
                l2 = complex_landmark
    if key == ord("e"):
        l1,l2 = None,None
   
stream.stop()
# plt.show()

# cleanup the camera and close any open windows
vs.stop()
cv2.destroyAllWindows()