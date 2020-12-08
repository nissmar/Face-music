from scipy.spatial import ConvexHull
from math import sqrt
import numpy as np
import cv2


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords
    
def detect_landmark(gray,outimg,detector,predictor):
    landmark = None
    ratio = len(outimg)/len(gray)
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

def compute_diameter(landmark):
    hull = ConvexHull(np.array(landmark))
    diameter = 0
    for i in hull.vertices:
        for j in hull.vertices:
            dist = sqrt(sum((np.array(landmark[i])-np.array(landmark[j]))**2))
            diameter = max(diameter,dist)
    return diameter

def normalize_landmark(landmark, transform_info = False):
    n = len(landmark)
    barycenter = sum(np.array(landmark))/n
    diameter = compute_diameter(landmark)
    normalized_landmark = ((np.array(landmark)-barycenter)/diameter).flatten()
    if transform_info:
        normalized_landmark = np.concatenate((np.array([diameter]), barycenter, normalized_landmark))
    return normalized_landmark

def np_to_complex(arr):
    arr = np.array([e[0] - 1j*e[1] for e in arr])
    return arr

def landmark_angle(land):
    leye = sum(land[42:48])/6
    reye = sum(land[36:42])/6
    return np.angle(leye-reye)