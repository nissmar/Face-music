import cv2
import dlib
import numpy as np
from imutils.video import VideoStream
import pickle
import imutils

from landmark_processing import normalize_landmark
from sound_manager import SoundManager


# CONFIG
MAX_ANGLE = 20
THRESHOLD_UP = 0.65
EYEBROW_THRESHOLD = 0.7


#SVM model
with open('mouth_svm.pickle','rb') as pickle_in:
    svm_mouth = pickle.load(pickle_in)
with open('sourcils_svm.pickle','rb') as pickle_in:
    svm_eyebrows = pickle.load(pickle_in)
with open('tilt_svm.pickle','rb') as pickle_in:
    svm_tilt = pickle.load(pickle_in)
with open('pan_svm.pickle','rb') as pickle_in:
    svm_pan = pickle.load(pickle_in)

manager = SoundManager()

# states for note playing
was_left = False
was_up = False
ready_to_replay = False

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


def evaluate_svm(landmark, regr):
    if landmark is None:
        return 0
    input = np.array([normalize_landmark(landmark)])
    return regr.predict(input)[0]


i=0

while True:
    i+=1
    frame = vs.read()
    frame = cv2.flip(frame,1)
    nframe = imutils.resize(frame, width=400)
    height, width, _ = frame.shape
    landmark = detect_shape(nframe,frame)
    if not(landmark is None):
        cv2.line(frame,(int(width/2), 0),(int(width/2), int(height)), (255, 0, 0))
        cv2.line(frame,(0,int(height*THRESHOLD_UP)),(int(width), int(height*THRESHOLD_UP)), (255, 0, 0))
        tilt, pan = evaluate_svm(landmark, svm_tilt), evaluate_svm(landmark, svm_pan)
        tilt = 1-min(max((tilt+MAX_ANGLE)/2/MAX_ANGLE,0),1)
        pan = 1-min(max((pan+MAX_ANGLE)/2/MAX_ANGLE,0),1)

        is_left = pan <= 0.5
        is_up = tilt <= THRESHOLD_UP
        
        cv2.circle(frame, (int(pan*width), int(tilt*height)), 10, (0, 255, 0), -1)

        if was_left != is_left:
            manager.play_note(is_left)
            ready_to_replay = False
        if (not is_up) and ready_to_replay:
            manager.play_note(is_left)
            ready_to_replay = False
        if (not was_up) and is_up:
            ready_to_replay = True
        was_left, was_up = is_left, is_up

    # detect eyebrows rise
    eyebrows_level = evaluate_svm(landmark, svm_eyebrows)
    if eyebrows_level > EYEBROW_THRESHOLD:
        manager.change_notes()

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

   
stream.stop()
# cleanup the camera and close any open windows
vs.stop()
cv2.destroyAllWindows()