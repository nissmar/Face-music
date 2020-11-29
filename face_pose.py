import dlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import glob 


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_68.dat')


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


def angles_from_title(title):
    if '-' in title:
        idx1 = title.index('-')
    if '+' in title:
        idx1 = min(idx1, title.index('+'))
    title_end = title[idx1+1:]
    if '-' in title_end:
        idx2 = idx1+1 + title_end.index('-')
    else:
        idx2 = idx1+1 + title_end.index('+')
    
    tilt = int(title[idx1:idx2])
    pan = int(title[idx2:-4])

    return tilt,pan

# img = cv2.imread('data/images/face1.jpeg',0) # reads image 'opencv-logo.png' as grayscale
# land = detect_shape(img)
# draw_land(img,land)
# plt.imshow(img, cmap='gray')
# plt.show()

img_dir = "data/HeadPoseImageDatabase/Person01" # Enter Directory of all images  
data_path = os.path.join(img_dir,'*g') 
files = glob.glob(data_path) 
for f1 in files[:10]: 
    img = cv2.imread(f1) 
    landmark = detect_shape(img)
    if not(landmark is None):
        tilt, pan = angles_from_title(f1)
        print(tilt,pan)
    else:
        print('nf')
    # plt.show()
