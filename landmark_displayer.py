import matplotlib.pyplot as plt
import numpy as np
from landmark_pickle import load_landmark

""" Displays landmark records. Use right and left arrow to navigate between frames """

ind = 0
fig, ax = plt.subplots()
ax.axis('equal')
rec = []


def split(arr):
    return [e[0] for e in arr], [-e[1] for e in arr]
    
def draw_landmark(ind):
    global rec
    plt.clf()
    x,y =split(rec[ind][1])
    plt.plot(x,y,'o')
    plt.draw()

def onclick(event):
    global ind, rec
    if event.key =="right":
        ind = min(ind+1, len(rec)-1)
    elif event.key =="left":
        ind = max(ind-1, 0)
    draw_landmark(ind)
    

def display_landmark(name):
    global rec
    rec = load_landmark(name)[0]
    draw_landmark(0)
    cid = fig.canvas.mpl_connect('key_press_event', onclick)
    plt.show()

display_landmark('sourcils')