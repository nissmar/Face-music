import matplotlib.pyplot as plt
import numpy as np
from landmark_pickle import load_landmark


def np_to_complex(arr):
    arr = np.array([e[0] - 1j*e[1] for e in arr])
    return arr

def normalized_to_complex(arr, normalize=True):
    arr = np.array([arr[2*i] - 1j*arr[2*i+1] for i in range(len(arr)//2)])
    if normalize:
        arr = (arr-arr.mean())/abs(arr[39]-arr[42]) # use the distance between the eyes as a reference
    return arr

def ratio(c0,cn,c):
    """ compute where c is on the [c0,cn] segment """
    cn -= c0
    c -= c0
    return (c.real*cn.real+c.imag*cn.imag)/abs(cn)/abs(cn)

def distance(c0,cn,c):
    x = abs(c-c0)
    y = abs(cn-c)
    return y/(x+y)
def mean_ratio(land1,land2,land):
    """ compute the mean ratio of land between land1 and land2 """
    return np.array([ratio(land1[j],land2[j],land[j]) for j in range(len(land))]).mean()

def plotc(arr):
    plt.plot(arr.real,arr.imag,'o')
    
def lat_angle(land,m,M):
    return (abs((land[28]-land[1])/(land[16]-land[1]))-m)/(M-m)

def vert_angle(land,m,M):
    return (abs( (land[30]-land[27]).imag) - m)/(M-m)

def left_eye(land,m,M):
    return (abs( (land[47]+land[46]-land[43]-land[44]).imag) - m)/(M-m)

def landmark_angle(land):
    leye = sum(land[42:48])/6
    reye = sum(land[36:42])/6
    return np.angle(leye-reye)

def rotate_landmark(land, angle):
    return [e*np.exp(1j*angle) for e in land]

if __name__ == "__main__":
    rec = load_landmark('turn')[0]
    print(len(rec))

    landmarks = [np_to_complex(land[1]) for land in rec]
    l1 = landmarks[0]
    l2 = landmarks[-1]

    means = [mean_ratio(l1,l2,land) for land in landmarks]
    plt.plot(means)
    plt.show()