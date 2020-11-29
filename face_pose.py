import dlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import glob 
import pickle
from sklearn import svm
from landmark_processing import normalize_landmark



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_68.dat')

## compute landmark
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords
    
def detect_shape(img):
    landmark = None
    rects = detector(img, 1) # rects contains all the faces detected
    for (i, rect) in enumerate(rects):
        shape = predictor(img, rect)
        landmark = shape_to_np(shape)
    return landmark

def draw_land(img):
    landmark = detect_shape(img)
    if not(landmark is None):
        for i in range(len(landmark)):
            x,y = landmark[i]
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

## extract pose from images
def angles_from_title(title):
    idx1 = 100
    if '-' in title:
        idx1 = title.index('-')
    if '+' in title:
        idx1 = min(idx1, title.index('+'))
    title_end = title[idx1+1:]
    if '-' in title_end:
        idx2 = idx1 + 1 + title_end.index('-')
    else:
        idx2 = idx1 + 1 + title_end.index('+')
    tilt = int(title[idx1:idx2])
    pan = int(title[idx2:-4])
    return tilt,pan

def pickle_out_pose():
    img_dirs = ["data/HeadPoseImageDatabase/Person0" + str(i) for i in range(1,10)] # Enter Directory of all images  
    img_dirs += ["data/HeadPoseImageDatabase/Person1" + str(i) for i in range(6)]
    landmarks, tilts, pans, persons = [], [], [], []

    for img_dir in img_dirs:
        print(img_dir)
        data_path = os.path.join(img_dir,'*g') 
        files = glob.glob(data_path) 
        for f1 in files: 
            img = cv2.imread(f1) 
            landmark = detect_shape(img)
            if not(landmark is None):
                landmarks.append(landmark)
                tilt, pan = angles_from_title(f1)
                tilts.append(tilt)
                pans.append(pan)
                persons.append(img_dir)
    
    with open(f"data/HeadPoseImageDatabase/landmarks_pose2.pickle", 'wb') as pickle_out:
        pickle.dump([landmarks,tilts,pans,persons], pickle_out)

with open('data/HeadPoseImageDatabase/landmarks_pose.pickle', 'rb') as pickle_in:
    data = pickle.load(pickle_in)

landmarks,tilts,pans,persons = data
landmarks = [normalize_landmark(e) for e in landmarks]


def picke_out_reg(landmarks,tilts,pans):
    regr = svm.SVR()
    regr.fit(landmarks,tilts)
    with open('tilt_svm.pickle','wb') as pickle_out:
        pickle.dump(regr, pickle_out)
    regr = svm.SVR()
    regr.fit(landmarks,pans)
    with open('pan_svm.pickle','wb') as pickle_out:
        pickle.dump(regr, pickle_out)


with open('tilt_svm.pickle','rb') as pickle_in:
    svm_tilt = pickle.load(pickle_in)
with open('pan_svm.pickle','rb') as pickle_in:
    svm_pan = pickle.load(pickle_in)
 
def test_SVR(landmarks,labels,svm):
    predicted_labels = svm.predict(landmarks)
    true_labels = np.asarray(labels)
    N = len(true_labels)
    return sum(list(map(lambda x: abs(x), list(true_labels-predicted_labels))))/N

print(test_SVR(landmarks,tilts,svm_tilt))
print(test_SVR(landmarks,pans,svm_pan))