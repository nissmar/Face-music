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
from explicit import np_to_complex, mean_ratio, lat_angle,vert_angle
from landmark_processing import normalize_landmark

#tensorflow model
dnn_model = keras.models.load_model('mouth_model')
#SVM model
with open('../svm/mouth_svm.pickle','rb') as pickle_in:
    svm_regr = load(pickle_in)
with open('../svm/tilt_svm.pickle','rb') as pickle_in:
    svm_tilt = load(pickle_in)
with open('../svm/pan_svm.pickle','rb') as pickle_in:
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




ready_for_record = False
is_capturing = False
stream = sd.OutputStream(channels=1, callback=callback,blocksize=1500)



while True:
    frame = vs.read()
    frame = cv2.flip(frame,1)
    nframe = imutils.resize(frame, width=400)
    landmark = detect_shape(nframe,frame)

    if is_capturing:
        is_capturing, frame = saver.poke(frame,landmark)
    cv2.imshow('Face music', frame)

    key = cv2.waitKey(100) & 0xFF
    if key == ord("q"):
        break
    if ready_for_record and not is_capturing and key == ord(" "):
        print("Starting to record for emotion:", saver.emotion_label)
        ready_for_record = False
        is_capturing = True
        saver.begin_record()
    if not ready_for_record and not is_capturing and key == ord(" "):
        ready_for_record = True
        print("Please type an emotion label corresponding to this record:")
        #emotion_label = input()
        emotion_label = "sourcils"
        saver.set_emotion_label(emotion_label)
    
   
stream.stop()
# cleanup the camera and close any open windows
vs.stop()
cv2.destroyAllWindows()