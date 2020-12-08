import cv2
import dlib
import numpy as np
from imutils.video import VideoStream
import pickle
import imutils

from landmark_processing import detect_landmark, normalize_landmark, landmark_angle, np_to_complex
from sound_manager import SoundManager


# CONFIG
MAX_ANGLE = 20
THRESHOLD_UP = 0.65
THRESHOLD_YAW = 0.25
SWITCHFRAME = 3 # wait between x frames to repeat order
DEAD_REGION = 0.5
EYEBROW_THRESHOLD = 0.48


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

def draw_instrument(height, width):
    h0 = int(height*(1-DEAD_REGION)/2)
    h1 = int(height-height*(1-DEAD_REGION)/2)
    w0 = int(width*(1-DEAD_REGION)/2)
    w1 = int(width-width*(1-DEAD_REGION)/2)
    wd = int(width/2)
    hd = int(height/2)
    height = int(height)
    widht = int(width)

    cv2.line(frame,(wd, 0),(wd, h0), (255, 0, 0))
    cv2.line(frame,(wd, h1),(wd, height), (255, 0, 0))
    cv2.line(frame,(0, hd),(w0, hd), (255, 0, 0))
    cv2.line(frame,(w1, hd),(width, hd), (255, 0, 0))
    
    cv2.line(frame,(w0, hd),(wd, h0), (255, 0, 0))
    cv2.line(frame,(wd, h0),(w1, hd), (255, 0, 0))
    cv2.line(frame,(w1, hd),(wd, h1), (255, 0, 0))
    cv2.line(frame,(wd, h1),(w0, hd), (255, 0, 0))




# variables
previous_region = 0
was_next = 0
was_prev = 0
mouth_open = 0
i=0
while True:
    i+=1
    frame = vs.read()
    frame = cv2.flip(frame,1)
    nframe = imutils.resize(frame, width=400)
    height, width, _ = frame.shape
    landmark = detect_landmark(cv2.cvtColor(nframe, cv2.COLOR_BGR2GRAY),frame,detector,predictor)
    if not(landmark is None):
        normalized_landmark = normalize_landmark(landmark)
        complex_landmark = np_to_complex(landmark)
        
        draw_instrument(height, width)

        tilt, pan = evaluate_svm(normalized_landmark, svm_tilt), evaluate_svm(normalized_landmark, svm_pan)
        tilt = 1-min(max((tilt+MAX_ANGLE)/2/MAX_ANGLE,0),1)
        pan = 1-min(max((pan+MAX_ANGLE)/2/MAX_ANGLE,0),1)
        cv2.circle(frame, (int(pan*width), int(tilt*height)), 10, (0, 255, 0), -1)

        # play note
        region = current_region(tilt,pan)
        if region != previous_region and region>0:
            manager.play_note(region%2)

        # detect eyebrows rise
        eyebrows_level = evaluate_svm(normalized_landmark, svm_eyebrows)
        if eyebrows_level > EYEBROW_THRESHOLD:
            manager.change_notes()

        # detect mouth opening 
        mouth_level = evaluate_svm(normalized_landmark, svm_mouth)
        if mouth_level > 0.3 and mouth_open==0:
            mouth_open = SWITCHFRAME
            manager.record_key()
            print('RECORDING')

        # detect face Yaw, change instrument 
        face_angle = landmark_angle(complex_landmark)
        if face_angle>THRESHOLD_YAW and was_prev==0:
            was_prev=SWITCHFRAME
            manager.change_mode(-1)
            print('Previous instrument')
        if face_angle<-THRESHOLD_YAW and was_next==0:
            was_next=SWITCHFRAME
            manager.change_mode(1)
            print('Next instrument')
 
        # variables changes
        previous_region = region
        was_next = max(0,was_next-1)
        was_prev = max(0,was_prev-1)
        mouth_open = max(0,mouth_open-1)
    # refresh sound_manager
    mode, is_recording, ready_for_record = manager.loop()

    # display mode
    cv2.putText(frame, mode, (width - 100, 35), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (255,0,0), thickness = 2)

    # show if recording with a green/red circle
    color = [(0,0,255),(0,165,255),(0,255,0)][2*is_recording+ready_for_record]
    text = ['Not recording','Ready for record','Recording'][2*is_recording+ready_for_record]
    cv2.circle(frame, (30,30), 20, color, -1)
    cv2.putText(frame, text, (60,35), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = color, thickness = 2)

    cv2.imshow('Face music', frame)

    key = cv2.waitKey(100) & 0xFF
    if key == ord("q"):
        break
    if key == ord("m"):
        manager.change_mode()
    if key == ord(' '):
        manager.record_key()
    if key == ord('d'):
        manager.delete_cur_mode_soundtrack()
    # keys to simulate notes
    if key == ord('a'):
        manager.play_note(True)
    if key == ord('p'):
        manager.play_note(False)

   
# cleanup the camera and close any open windows
vs.stop()
cv2.destroyAllWindows()