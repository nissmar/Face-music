import sounddevice as sd
import soundfile as sf

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.stats import gamma

# PARAMS

FREQUENCY = 440 # goal
SAMPLERATE = 44100 # may depend on the computer



def adjust(variable, target, d=20):
    """ variable <- target with a step of at most d """
    if variable>d+target:
        variable -= d
    elif variable<target-d:
        variable += d
    else:
        variable = target
    return variable

# WAVE GENERATION

def sin_transition(freq1, freq2, frames, samplerate=SAMPLERATE):
    """sine function with variable frequency"""
    t = np.arange(frames)/ samplerate
    G = (freq2*t*t - freq1*(t[-1]-t)*(t[-1]-t))/2/t[-1] + t[-1]/2*freq1 # integrates (1-t)f1 + t*f2
    return np.sin(2 * np.pi * G)

def sin_n(freq, frames, samplerate=SAMPLERATE):
    """sine function with phase"""
    t = np.arange(frames)/ samplerate
    return np.sin(2 * np.pi * freq * t)

def square(freq, frames, samplerate=SAMPLERATE):
    t = np.arange(frames)/ samplerate
    return 2*( ((t%(1.0/freq))<(1.0/freq)/2) - 0.5)

def triangle(freq, frames, samplerate=SAMPLERATE):
    t = np.arange(frames)/ samplerate
    return 2*( np.abs((signal.sawtooth(2*np.pi * freq * t-np.pi/2))) - 0.5 )

def envelope(x, frames, samplerate=SAMPLERATE):
    t = np.arange(frames)/ samplerate
    return np.sin((t/t.max())**x*np.pi)

# GOOD VIBES
def good1(freq,frames,samplerate=SAMPLERATE):
    env = envelope(1.4,frames)**2
    env2 = envelope(0.2,frames)
    wave = sin_n(freq,frames)
    wave2 = triangle(freq,frames)
    return env*wave +env2*wave2

def good1_chord(freq,frames,samplerate=SAMPLERATE):
    return good1(freq,frames)+good1(freq*1.5,frames)/2+good1(freq*2,frames)/3 

def onclick(event):
    frames = 15000
    x = min(16*event.x/frames,2)
    y = min(event.y/800,1)
    wave = good1_chord(event.x,frames)
    
    plt.clf()
    plt.plot(wave)
    plt.draw()
    sd.play(wave)
 
# interaction
if __name__ == "__main__":
    fig, ax = plt.subplots()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    # frames = 100000
    # L=[]
    # func = sin_n
    # f1 = 800
    # f2 =1600
    # PHASE = 0

    # # plt.plot(envelope(0.2,frames))
    # # plt.plot(envelope(0.8,frames))
    # # plt.plot(envelope(1.2,frames))
    # # plt.plot(sin_n(400,frames))
    # # plt.plot(sin_transition(400,800,frames))
    # # plt.plot(triangle(400,frames))
    # plt.show()
