from scipy.spatial import ConvexHull
from math import sqrt
import numpy as np

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