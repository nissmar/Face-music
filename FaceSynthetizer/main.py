import cv2
import dlib
import numpy as np
from imutils.video import VideoStream
import pickle
import imutils

from landmark_processing import detect_landmark, normalize_landmark, landmark_angle, np_to_complex
from sound_manager import SoundManager

import sounddevice as sd
from fluid_manager import callback, set_freq, set_mix,set_harmonic


# CONFIG
MAX_ANGLE = 20
THRESHOLD_MOUTH = 0.68
THRESHOLD_YAW = 0.35
DEAD_REGION = 0.3
EYEBROW_THRESHOLD = 0.6
IS_CALIBRATING = 0
MODE = "SYNTH" # SYNTH ou FLUID


#SVM model
with open('mouth_svm.pickle','rb') as pickle_in:
    svm_mouth = pickle.load(pickle_in)
with open('sourcils_svm.pickle','rb') as pickle_in:
    svm_eyebrows = pickle.load(pickle_in)
with open('tilt_svm.pickle','rb') as pickle_in:
    svm_tilt = pickle.load(pickle_in)
with open('pan_svm.pickle','rb') as pickle_in:
    svm_pan = pickle.load(pickle_in)

def evaluate_svm(normalized_landmark, regr):
    if landmark is None:
        return 0
    input = np.array([normalized_landmark])
    return regr.predict(input)[0]


manager = SoundManager()

# Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_68.dat')

vs = VideoStream(src=0).start()


def current_region(x,y):
    # 1       2
    # DEAD_ZONE
    # 3      4
    if abs(x-0.5)+abs(y-0.5)<DEAD_REGION/2:
        return 0 
    if x>0.5:
        return 4 if y>0.5 else 3
    return 2 if y>0.5 else 1

def draw_synth(height, off):
    w0 = int(height*(1-DEAD_REGION)/2)
    w1 = int(height-height*(1-DEAD_REGION)/2)
    wd = int(height/2)
    
    height = int(height)
    off = int(off)

    cv2.line(frame,(off, 0),(off, height), (255, 0, 0))
    cv2.line(frame,(height+off, 0),(height+off, height), (255, 0, 0))

    cv2.line(frame,(wd+off, 0),(wd+off, w0), (255, 0, 0))
    cv2.line(frame,(wd+off, w1),(wd+off, height), (255, 0, 0))
    cv2.line(frame,(off, wd),(w0+off, wd), (255, 0, 0))
    cv2.line(frame,(w1+off, wd),(height+off, wd), (255, 0, 0))

    cv2.circle(frame,(wd+off,wd), int(w1-w0)//2, (255,0,0))

def display_measure(img, mu):
    """Display position in measure as a slider. mu in [0,1]"""
    width = img.shape[1]
    x1,x2,y1,y2 = width-100,width-20,60,90
    
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 1)
    x1,y1,x2,y2 = x1+1,y1+1,int(x1+mu*(x2-x1-2)),y2-1
    cv2.rectangle(img,(x1,y1), (x2,y2),(255,0,0), -1)

def display_number_of_records(img, n):
    """Display number of records for current mode"""
    width = img.shape[1]
    radius = 5
    x,y = width-100+radius,110
    for _ in range(n):
        cv2.circle(img, (x,y), radius, (255,0,0), -1)
        x += 3*radius


def calibrate(mouth_level,eyebrows_level, frame):
    global IS_CALIBRATING, EYEBROW_THRESHOLD, THRESHOLD_MOUTH

    text = ['Calibrating', 'Calibrating Mouth','Calibrating eyebrows','Done'][IS_CALIBRATING-1]
    cv2.putText(frame, text, (60,70), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = color, thickness = 2)
    if IS_CALIBRATING==1:
        print('CALIBRATING...')
        THRESHOLD_MOUTH = 0
        EYEBROW_THRESHOLD = 0
        IS_CALIBRATING+=1
    elif IS_CALIBRATING==2:
        THRESHOLD_MOUTH = max(THRESHOLD_MOUTH,0.9*mouth_level)
    elif IS_CALIBRATING==3:
        EYEBROW_THRESHOLD = max(EYEBROW_THRESHOLD,0.95*eyebrows_level)
    elif IS_CALIBRATING==4:
        print(' mouth level:',THRESHOLD_MOUTH)
        print(' eyebrows level:',EYEBROW_THRESHOLD)
        print('  ...DONE')
        IS_CALIBRATING = 0


# variables
previous_region = 0
previous_angle = 0 #0 =no 1= right 2 = left 3=switch
face_angle = 0
brows_raised = False
fluid_initialized=False
mouth_opened = False
i=0

# fluid
stream = sd.OutputStream(channels=1, callback=callback,blocksize=1500)

while True:
    i+=1
    frame = vs.read()
    frame = cv2.flip(frame,1)
    nframe = imutils.resize(frame, width=400)
    height, width, _ = frame.shape
    offset = (width-height)/2
    landmark = detect_landmark(cv2.cvtColor(nframe, cv2.COLOR_BGR2GRAY),frame,detector,predictor)
    if not(landmark is None):
        normalized_landmark = normalize_landmark(landmark)
        complex_landmark = np_to_complex(landmark)
        
        tilt, pan = evaluate_svm(normalized_landmark, svm_tilt), evaluate_svm(normalized_landmark, svm_pan)
        tilt = 1-min(max((tilt+MAX_ANGLE)/2/MAX_ANGLE,0),1)
        pan = 1-min(max((pan+MAX_ANGLE)/2/MAX_ANGLE,0),1)

        # detect face Yaw, change instrument 
        x = landmark_angle(complex_landmark)
        if x>THRESHOLD_YAW:
            face_angle=1
        elif x<-THRESHOLD_YAW:
            face_angle=2
        else:
            face_angle=0

        eyebrows_level = evaluate_svm(normalized_landmark, svm_eyebrows)

        mouth_level = evaluate_svm(normalized_landmark, svm_mouth)

        if IS_CALIBRATING>0:
            calibrate(mouth_level, eyebrows_level, frame)
        elif MODE=="SYNTH":
            draw_synth(height, offset)
            cv2.circle(frame, (int(pan*height+offset), int(tilt*height)), 10, (0, 255, 0), -1)

            # detect face Yaw, change instrument 
            if face_angle==1 and previous_angle==0:
                manager.change_mode(-1)
                # MODE = "FLUID"
            elif face_angle==2 and previous_angle==0:
                manager.change_mode(1)

            # play note
            region = current_region(tilt,pan)
            if region != previous_region and region>0:
                manager.play_note(region-1)
            previous_region = region

            # detect mouth opening 
            if mouth_level > THRESHOLD_MOUTH and face_angle==0:
                if not(mouth_opened):
                    mouth_opened=True
                    manager.record_key()
            else:
                mouth_opened=False
        
            # detect eyebrows rise
            if eyebrows_level > EYEBROW_THRESHOLD and not(mouth_level>0.5*THRESHOLD_MOUTH) and region==0 and not(brows_raised):
                manager.delete_cur_mode_last_record()
                brows_raised = True
            else:
                brows_raised = False

        else:
            cv2.circle(frame, (int(pan*height+offset), int(tilt*height)), 10, (0, 255, 0), -1)

            if face_angle==0 and not(fluid_initialized): #initialize fluid
                fluid_initialized = True
                stream.start()
            if fluid_initialized:

                # switch mode if head is angled
                if face_angle>0 and previous_angle==0:
                    stream.stop()
                    MODE = "SYNTH"

                set_freq(220*(1+0.5*pan))

                set_mix(mouth_level*0.5)

                set_harmonic(tilt)





    
    previous_angle = face_angle

    # refresh sound_manager
    mode, number_of_records, is_recording, ready_for_record, pos_in_measure = manager.loop()

    # display mode
    cv2.putText(frame, mode, (width - 100, 35), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (255,0,0), thickness = 2)

    # display number of records for current mode
    display_number_of_records(frame, number_of_records)

    # show if recording with a green/red circle
    color = [(0,0,255),(0,165,255),(0,255,0)][2*is_recording+ready_for_record]
    text = ['Not recording','Ready for record','Recording'][2*is_recording+ready_for_record]
    cv2.circle(frame, (30,30), 20, color, -1)
    cv2.putText(frame, text, (60,35), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = color, thickness = 2)

    # display measure slider
    if pos_in_measure > 0:
        display_measure(frame, pos_in_measure)

    cv2.imshow('Face music', frame)

    key = cv2.waitKey(100) & 0xFF
    if key == ord("q"):
        break
    if key == ord('c'): # to calibrate
        IS_CALIBRATING+=1
    if key == ord("m"):
        manager.change_mode()
    if key == ord(' '):
        manager.record_key()
    if key == ord('d'):
        manager.delete_cur_mode_last_record()
    # keys to simulate notes
    if key == ord('a'):
        manager.play_note(True)
    if key == ord('p'):
        manager.play_note(False)

   
# cleanup the camera and close any open windows
vs.stop()
cv2.destroyAllWindows()