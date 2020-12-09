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
def square_signal(x,T):
    return (x%T)<T/2

def mix(freq, frames, samplerate=SAMPLERATE):
    """mix of a sine and a triangle function with 1.5 freq """
    global PHASE
    t = np.arange(frames)/ samplerate

    square = square_signal(t + PHASE/2/np.pi/freq,1/freq)-0.5
    sine = np.sin(2 * np.pi * freq * t+PHASE)/2
    sinh = np.sin(2 * np.pi * 1.5*freq * t + 1.5*PHASE)/2

    PHASE += 2 * np.pi * (freq * (t[-1]+t[1])) # phase shift
    return ((1-MIX_FAC)*2*((1-HARMONIC_FAC)*sine+HARMONIC_FAC*sinh) + MIX_FAC*square).reshape(-1,1)

# SOUNDDEVICE
def callback(outdata, frames, time, status):
    global FREQUENCY,FREQUENCY_PLAYING, WAVE
    # FREQUENCY_PLAYING=FREQUENCY
    WAVE.append(mix(FREQUENCY, frames))
    outdata[:] = WAVE[-1] 
    # FREQUENCY_PLAYING=adjust(FREQUENCY_PLAYING,FREQUENCY,100)


def set_freq(freq):
    global FREQUENCY
    FREQUENCY = freq

def set_mix(mix,ad=True):
    global MIX_FAC
    if ad:
        MIX_FAC = adjust(MIX_FAC,mix,0.1)
    else:
        MIX_FAC=mix


def set_harmonic(mix,ad=True):
    global HARMONIC_FAC
    if ad:
        HARMONIC_FAC = adjust(HARMONIC_FAC,mix,0.1)
    else:
        HARMONIC_FAC=mix

def get_wave():
   return WAVE


# def onclick(event):
#     global FREQUENCY
#     FREQUENCY = event.x
#     plt.clf()
#     plt.plot(WAVE)
#     plt.draw()
# # interaction
# if __name__ == "__main__":
#     MIX_FAC = 0.5

#     stream = sd.OutputStream(channels=1, callback=callback,blocksize=0)
#     stream.start()

#     fig, ax = plt.subplots()
#     cid = fig.canvas.mpl_connect('button_press_event', onclick)
#     plt.show()
#     stream.stop()

# frames = 100
# L=[]
# outdata=[]
# set_mix(0.5,False)
# set_harmonic(0.5,False)

# f1 = 800
# f2 =500
# PHASE = 0
# set_freq(f1)
# callback(outdata,frames,0,0)
# set_freq(f2)
# print(PHASE)
# print(len(WAVE))

# callback(outdata,frames,0,0)
# set_freq(f1)
# print(PHASE)
# print(len(WAVE))


# callback(outdata,frames,0,0)
# plt.plot(np.concatenate(WAVE))

# plt.show()
