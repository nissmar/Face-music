import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# PARAMS

SAMPLERATE = 44100 # may depend on the computer
FREQUENCY_PLAYING = 440 # current frequency
FREQUENCY = 440 # goal
MIX_FAC = 0
HARMONIC_FAC = 0
WAVE = []
PHASE = 0 # phase



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
    global PHASE

    t = np.arange(frames)/ samplerate
    G = (freq2*t*t - freq1*(t[-1]-t)*(t[-1]-t))/2/t[-1] + t[-1]/2*freq1 # integrates (1-t)f1 + t*f2
    out = np.sin(2 * np.pi * G + PHASE).reshape(-1,1)    
    PHASE += 2 * np.pi * (G[-1] + freq2*t[1]) # phase shift
    return out

def sin_n(freq, frames, samplerate=SAMPLERATE):
    """sine function with phase"""
    global PHASE
    t = np.arange(frames)/ samplerate
    out = np.sin(2 * np.pi * freq * t + PHASE).reshape(-1,1)    
    PHASE += 2 * np.pi * (freq * (t[-1]+t[1])) # phase shift
    return out

def square_signal(x,T):
    return (x%T)<T/2
def mix(freq, fac, fach, frames, samplerate=SAMPLERATE):
    """mix of a sine and a triangle function with 1.5 freq """
    global PHASE
    t = np.arange(frames)/ samplerate
    # triangle = np.abs((signal.sawtooth(2*np.pi * freq * t + PHASE-np.pi/2)))-0.5
    square = square_signal(t + PHASE/2/np.pi/freq,1/freq)-0.5
    sin = np.sin(2 * np.pi * freq * t + PHASE)/2
    sinh = np.sin(2 * np.pi * 1.5*freq * t + 1.5*PHASE)/2

    PHASE += 2 * np.pi * (freq * (t[-1]+t[1])) # phase shift
    return ((1-fac)*2*((1-HARMONIC_FAC)*sin+HARMONIC_FAC*sinh)+fac*square).reshape(-1,1)

# SOUNDDEVICE
def callback(outdata, frames, time, status):
    global FREQUENCY,FREQUENCY_PLAYING, WAVE

    # standard sine
    # WAVE = sin_transition(FREQUENCY_PLAYING, FREQUENCY, frames)
    outdata[:] = mix(FREQUENCY_PLAYING, MIX_FAC, HARMONIC_FAC, frames)

    WAVE.append(outdata)

    FREQUENCY_PLAYING=adjust(FREQUENCY_PLAYING,FREQUENCY,100)


def onclick(event):
    global FREQUENCY
    FREQUENCY = event.x
    plt.clf()
    plt.plot(WAVE)
    plt.draw()
    
def set_freq(freq):
    global FREQUENCY
    FREQUENCY = freq

def set_mix(mix):
    global MIX_FAC
    MIX_FAC = adjust(MIX_FAC,mix,0.1)

def set_harmonic(mix):
    global HARMONIC_FAC
    HARMONIC_FAC = adjust(HARMONIC_FAC,mix,0.1)


def get_wave():
   return WAVE

# interaction
if __name__ == "__main__":
    MIX_FAC = 0.5

    stream = sd.OutputStream(channels=1, callback=callback,blocksize=0)
    stream.start()

    fig, ax = plt.subplots()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    stream.stop()

# frames = 100
# L=[]
# func = sin_n
# f1 = 800
# f2 =1600
# PHASE = 0

# L.append(mix(f1,0.5, frames))
# L.append(mix(f2,0.5, frames))
# plt.plot(np.concatenate(L),'-')

# plt.show()
